#!/usr/bin/env python3
"""Fresh matched per-call timings for two eight-L4 NCCL-compatible libraries.

Each backend gets the same client, environment, exact shapes and layouts.
Both use eight processes/one GPU each. Every warmup and timed call verifies
full output/guards. Preparation and verification are outside CUDA-event timing.
All repetitions are retained; backend order reverses in alternating jobs.
"""
import argparse
import csv
import fcntl
import json
import math
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
import time

from compare_local8_reference import audit_routes, samples, sha
import local8_process

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'integration/nccl_lite'))
import backend_profile


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--baseline', type=Path, required=True)
    p.add_argument('--candidate', type=Path, required=True)
    p.add_argument('--client', type=Path, required=True)
    p.add_argument('--real-nccl', type=Path, default=ROOT / 'build/nccl-2.28.9/lib/libnccl.so.2')
    p.add_argument('--max-bytes', type=int, default=1 << 30)
    p.add_argument('--warmup', type=int, default=5)
    p.add_argument('--iterations', type=int, default=10)
    p.add_argument('--repeats', type=int, default=3)
    p.add_argument('--jobs', type=int, default=2)
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    if not 1024 <= args.max_bytes <= 1 << 30 or args.max_bytes & (args.max_bytes - 1):
        p.error('max-bytes must be a power of two in 1 KiB–1 GiB')
    if min(args.warmup, args.iterations, args.repeats, args.jobs) < 1:
        p.error('positive warmup/iteration/repeat/job counts required')
    if os.environ.get('LD_PRELOAD'):
        p.error('unset LD_PRELOAD before validation')
    lock = open('/tmp/uccl-lite-integration-wxz-l4.lock', 'a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid',
                                '--format=csv,noheader,nounits'], text=True).strip():
        raise RuntimeError('GPU already occupied')
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    client = out / 'reference_timing_check'
    shutil.copy2(args.client, client)
    for name in ['compare_local8_backends.py', 'compare_local8_reference.py', 'local8_process.py']:
        shutil.copy2(Path(__file__).with_name(name), out / name)
    real = args.real_nccl.resolve(strict=True)
    env = {k: v for k, v in os.environ.items()
           if k in ('PATH', 'HOME', 'USER', 'LOGNAME', 'LANG', 'TMPDIR',
                    'CUDA_VISIBLE_DEVICES', 'CUDA_DEVICE_ORDER')}
    env.update(NCCL_NET='Socket', NCCL_NET_PLUGIN='none', NCCL_PROFILER_PLUGIN='none',
               NCCL_IB_DISABLE='1', UCCL_LITE_MSCCLPP_CONTROL_ROOT='127.0.0.1',
               LD_LIBRARY_PATH=str(real.parent) + ':/usr/local/cuda-12.9/lib64')
    env.update(backend_profile.local_packet_environment())
    sizes = [1 << power for power in range(10, args.max_bytes.bit_length())]
    shapes = [(op, size // (2 if op == 3 else 16)) for op in (1, 2, 3) for size in sizes]
    manifest = dict(timing=__doc__, hardware='wxz-l4, eight L4s, BF16, one GPU/process',
                    baseline_revision='pre-refactor qualified .so; identity recorded below',
                    source_revision=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                    environment=env, sizes=sizes, client_sha256=sha(client),
                    warmup=args.warmup, iterations=args.iterations, repeats=args.repeats,
                    jobs=args.jobs, backends={}, commands=[], results={})
    commands = {}
    for name, source in [('baseline', args.baseline), ('candidate', args.candidate)]:
        folder = out / name
        folder.mkdir()
        library = folder / 'libuccl_lite_nccl_local8.so'
        shutil.copy2(source, library)
        (folder / 'strict').mkdir()
        (folder / 'strict/libnccl.so.2').symlink_to('../' + library.name)
        for helper in ['backend_profile.py', 'dropin_profile.py']:
            shutil.copy2(ROOT / 'integration/nccl_lite' / helper, folder / helper)
        dependencies = subprocess.check_output(['ldd', str(library)], text=True)
        if 'libnccl.so' in dependencies or 'not found' in dependencies:
            raise RuntimeError('native NCCL dependency or unresolved library: ' + name)
        (folder / 'dependencies.txt').write_text(dependencies)
        launcher = ['python3', '-B', str(folder / 'backend_profile.py')]
        options = ['--local8', '--library', str(library), '--real-nccl', str(real)]
        record = json.loads(subprocess.check_output(launcher + ['probe', *options], env=env, text=True))
        (folder / 'host0.json').write_text(json.dumps(record, indent=2) + '\n')
        profile = folder / 'candidate.tsv'
        backend_profile.write_candidate(profile, [record], shapes, local8=True)
        manifest['backends'][name] = dict(library_sha256=sha(library), profile_sha256=sha(profile),
                                          host_identity=record['identity'])
        commands[name] = ['mpirun', '--timeout', '1700', '-np', '8', '--bind-to', 'none',
                          '--map-by', 'slot', '--mca', 'pml', 'ob1', '--mca', 'btl', 'self,vader,tcp',
                          *launcher, 'launch', *options, '--profile', str(profile), '--', str(client)]
    # Only the library hash may differ in the recorded host/software/config.
    identities = [json.loads((out / name / 'host0.json').read_text())['details']
                  for name in ['baseline', 'candidate']]
    for identity in identities:
        identity.pop('library_sha256')
    if identities[0] != identities[1]:
        raise RuntimeError('baseline/candidate runtime configurations differ')

    def save():
        (out / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')

    save()
    for op, operation in [('ar', 3), ('ag', 1), ('rs', 2)]:
        for job in range(args.jobs):
            layouts = ['out', 'in'] if job % 2 == 0 else ['in', 'out']
            order = ['baseline', 'candidate'] if job % 2 == 0 else ['candidate', 'baseline']
            for layout in layouts:
                for name in order:
                    label = f'{op}_{layout}_{job}_{name}'
                    command = commands[name] + [op, layout, '1024', str(args.max_bytes),
                                                str(args.warmup), str(args.iterations), str(args.repeats)]
                    entry = dict(label=label, argv=command, started=time.time())
                    manifest['commands'].append(entry)
                    save()
                    print('Running', label, flush=True)
                    path = out / (label + '.log')
                    with path.open('x') as log:
                        child = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT,
                                                 start_new_session=True)
                        try:
                            status = local8_process.wait(child, 1800, log_path=path,
                                                          affinity_path=out / (label + '.affinity.json'))
                        except BaseException:
                            entry.update(returncode=child.returncode, interrupted=True)
                            save()
                            raise
                    entry.update(returncode=status, elapsed=time.time() - entry['started'])
                    save()
                    if status:
                        raise RuntimeError(label + ' failed; inspect retained log')
                    text = path.read_text()
                    audit_routes(text, operation, sizes, args.repeats * (args.warmup + args.iterations))
                    if args.max_bytes == 1 << 30 and not (out / (label + '.affinity.json')).is_file():
                        raise RuntimeError('missing runtime worker-isolation audit')
                    manifest['results'][label] = samples(text, sizes, 'candidate', args.repeats, args.iterations)
                    save()
    rows = []
    for op in ['ar', 'ag', 'rs']:
        for size in sizes:
            for layout in ['out', 'in']:
                values = {name: statistics.median(manifest['results'][f'{op}_{layout}_{job}_{name}'][size]
                                                  for job in range(args.jobs))
                          for name in ['baseline', 'candidate']}
                rows.append(dict(operation=op, bytes=size, layout=layout, baseline_us=values['baseline'],
                                 candidate_us=values['candidate'], ratio=values['candidate']/values['baseline']))
    with (out / 'comparison.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)
    summary = {}
    for op in ['ar', 'ag', 'rs']:
        ratios = [r['ratio'] for r in rows if r['operation'] == op]
        summary[op] = dict(points=len(ratios), median_ratio=statistics.median(ratios),
                           geomean_ratio=math.exp(statistics.mean(math.log(r) for r in ratios)),
                           worst_ratio=max(ratios), over_5_percent=sum(r > 1.05 for r in ratios))
    (out / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
