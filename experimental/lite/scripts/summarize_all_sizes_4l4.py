#!/usr/bin/env python3
"""Validate complete paired sweeps and publish small/medium/large views."""
import collections
import csv
import math
from pathlib import Path
from statistics import mean, median

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

root = Path(__file__).resolve().parents[1]
data = root / 'benchmarks/data/small_medium_4l4_20261113'
new = {'mini-host', 'scatter-hier-ll', 'push-grid', 'scatter-grid'}

def save(name, rows):
    with (data/name).open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader(); w.writerows(rows)

def read(log, operations, sizes, iterations, fixed=None, baseline="nccl"):
    calls = collections.defaultdict(lambda: collections.defaultdict(dict))
    for line in (data/log).read_text().splitlines():
        if line.startswith('# sample,'):
            _, op, size, backend, run, call, us = line.split(',')
            value = float(us)
            assert math.isfinite(value) and value > 0
            sample = calls[op, int(size), backend][int(run)]
            assert int(call) not in sample
            sample[int(call)] = value
    assert {(o,s) for o,s,_ in calls} == {(o,s) for o in operations for s in sizes}
    candidates, selected = [], []
    for op in operations:
        for size in sizes:
            stats = {}
            for (operation, extent, backend), runs in calls.items():
                if (operation, extent) != (op,size): continue
                assert set(runs) == {1,2,3}
                assert all(sorted(v) == list(range(1, iterations+1)) for v in runs.values())
                means = [mean(v.values()) for v in runs.values()]
                stats[backend] = (median(means), min(means), max(means))
                candidates.append(dict(operation=op, bytes=size, backend=backend,
                                       median_us=median(means), min_us=min(means), max_us=max(means)))
            assert baseline in stats
            best = fixed or min((b for b in stats if b!=baseline), key=lambda b:stats[b][0])
            prior = min((stats[b][0] for b in stats if b!=baseline and b not in new))
            selected.append(dict(operation=op, bytes=size, baseline=baseline, uccl_path=best,
                                 uccl_median_us=stats[best][0], uccl_min_us=stats[best][1],
                                 uccl_max_us=stats[best][2], nccl_median_us=stats[baseline][0],
                                 nccl_min_us=stats[baseline][1], nccl_max_us=stats[baseline][2],
                                 prior_best_us=prior, speedup=stats[baseline][0]/stats[best][0]))
    print(log, 'validated timed calls:', sum(len(v) for runs in calls.values() for v in runs.values()))
    return candidates, selected

agc, ag = read('ag_final.log', ['all_gather'], [16<<i for i in range(27)], 10)
rsc, rs = read('rs_final.log', ['reduce_scatter'], [16<<i for i in range(27)], 10)
rc, resident = read('resident_doorbell_final.log', ['all_gather','reduce_scatter','all_reduce'],
                    [16<<i for i in range(17)], 20, 'resident')
save('ag_candidates.csv', agc); save('ag_selected.csv', ag)
save('rs_candidates.csv', rsc); save('rs_selected.csv', rs)
save('resident_candidates.csv', rc); save('resident_selected.csv', resident)
gc, graph_control = read('resident_graph_final.log', ['all_gather','reduce_scatter','all_reduce'],
                         [16<<i for i in range(17)], 20, 'resident', 'nccl-graph')
save('resident_graph_candidates.csv', gc); save('resident_graph_selected.csv', graph_control)
with (data/'ar_selected.csv').open() as f:
    ar = list(csv.DictReader(f))
for row in ar:
    row['bytes'] = int(row['bytes'])
    for key in ['uccl_median_us','nccl_median_us','uccl_min_us','uccl_max_us','nccl_min_us','nccl_max_us']:
        row[key] = float(row[key])
assert [r['bytes'] for r in ar] == [1024<<i for i in range(21)]

def plot(rows, name, timer, factor):
    fig, axes = plt.subplots(1, 3, figsize=(14,4))
    small = [r for r in rows if r['bytes'] <= 65536]
    medium = [r for r in rows if 131072 <= r['bytes'] <= 16777216]
    large = [r for r in rows if r['bytes'] >= 33554432]
    for key, label in [('uccl_median_us','UCCL Lite selected'),('nccl_median_us','NCCL 2.28.7')]:
        axes[0].plot([r['bytes']/1024 for r in small], [r[key] for r in small], 'o-', label=label)
        axes[1].plot([r['bytes']/2**20 for r in medium], [r[key] for r in medium], 'o-', label=label)
        axes[2].plot([r['bytes']/2**20 for r in large],
                     [factor*r['bytes']/r[key]/1000 for r in large], 'o-', label=label)
    for ax in axes:
        ax.set_xscale('log', base=2); ax.grid(alpha=.3); ax.legend(fontsize=8)
    axes[0].set(xlabel='Full tensor/GPU (KiB)', ylabel='Latency (µs)', title='Small')
    axes[1].set(xlabel='Full tensor/GPU (MiB)', ylabel='Latency (µs)', title='Medium')
    axes[1].set_yscale('log', base=2)
    axes[2].set(xlabel='Full tensor/GPU (MiB)', ylabel='Bus bandwidth (GB/s)', title='Large')
    fig.suptitle(f'4×L4 FP32 {name} · {timer} · median of 3 run means')
    fig.tight_layout()
    fig.savefig(root/f'docs/assets/collectives-4l4-{name.lower()}-all-sizes.svg')
    fig.savefig(root/f'build/{name.lower()}-all-sizes.png', dpi=100)
plot(ar, 'AllReduce', '4 processes, host timer', 1.5)
plot(ag, 'AllGather', '1 process, eager GPU events', .75)
plot(rs, 'ReduceScatter', '1 process, eager GPU events', .75)
fig, axes = plt.subplots(1,3,figsize=(14,4))
for ax, op, label in zip(axes, ['all_reduce','all_gather','reduce_scatter'], ['AllReduce','AllGather','ReduceScatter']):
    rows = [r for r in resident if r['operation']==op]
    for key, title in [('uccl_median_us','Prepared resident'),('nccl_median_us','NCCL 2.28.7')]:
        ax.plot([r['bytes']/1024 for r in rows], [r[key] for r in rows], 'o-', label=title)
    ax.set_xscale('log',base=2); ax.set_yscale('log',base=2)
    ax.set(title=label,xlabel='Full tensor/GPU (KiB)',ylabel='Latency (µs)')
    ax.grid(alpha=.3); ax.legend(fontsize=8)
fig.suptitle('Experimental resident mode · input-ready, one-process host completion · preparation excluded')
fig.tight_layout(); fig.savefig(root/'docs/assets/collectives-4l4-resident-small-medium.svg')
fig.savefig(root/'build/resident-small-medium.png',dpi=100)

def size_name(n):
    return f'{n//2**30} GiB' if n>=2**30 else f'{n//2**20} MiB' if n>=2**20 else f'{n//1024} KiB' if n>=1024 else f'{n} B'
def cell(row):
    a,b=row['uccl_median_us'],row['nccl_median_us']
    return f'{a:,.2f} / {b:,.2f}'+(' †' if a>b else '')
lookup = [{r['bytes']:r for r in rows} for rows in [ar,ag,rs]]
lines=['| Full bytes/GPU | AllReduce | AllGather | ReduceScatter |', '| ---: | ---: | ---: | ---: |']
for n in [1024<<i for i in range(21)]:
    lines.append('| '+size_name(n)+' | '+' | '.join(cell(v[n]) for v in lookup)+' |')
(data/'ordinary_table.md').write_text('\n'.join(lines)+'\n')
rlookup={op:{r['bytes']:r for r in resident if r['operation']==op} for op in ['all_reduce','all_gather','reduce_scatter']}
lines=['| Full bytes/GPU | Resident AR / NCCL | Resident AG / NCCL | Resident RS / NCCL |', '| ---: | ---: | ---: | ---: |']
for n in [16,1024,16384,65536,262144,1048576]:
    lines.append('| '+size_name(n)+' | '+' | '.join(cell(rlookup[op][n]) for op in rlookup)+' |')
(data/'resident_table.md').write_text('\n'.join(lines)+'\n')
print('AG medium improvements:')
for row in ag:
    if row['bytes'] in [2**20,2**21,2**22,2**23,2**24]: print(row)

# Focused views for the requested 1 KiB–1 MiB range. Keep execution contracts
# in separate figures and retain all 11 sizes, including slower medians.
def focus_plot(groups, filename, title, resident_mode=False):
    from matplotlib.ticker import ScalarFormatter
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.6))
    sizes = [1024 << i for i in range(11)]
    for ax, rows, label in zip(axes, groups, ['AllReduce', 'AllGather', 'ReduceScatter']):
        rows = [r for r in rows if 1024 <= r['bytes'] <= 1048576]
        assert [r['bytes'] for r in rows] == sizes
        for key, series, style in [
            ('uccl_median_us', 'Prepared resident' if resident_mode else 'UCCL Lite selected', 'o-'),
            ('nccl_median_us', 'NCCL 2.28.7', 's--'),
        ]:
            ax.plot([r['bytes']/1024 for r in rows], [r[key] for r in rows],
                    style, markersize=4, linewidth=1.7, label=series)
        scope = '1 process · input-ready host timing' if resident_mode else (
            '4 processes · host timing' if label == 'AllReduce' else '1 process · eager GPU events')
        ax.set_title(label + '\n' + scope, fontsize=10)
        ax.set_xscale('log', base=2); ax.set_yscale('log', base=2)
        ax.set_xticks([n/1024 for n in sizes])
        ax.set_xticklabels([str(n//1024) for n in sizes], rotation=45, ha='right', fontsize=8)
        ax.yaxis.set_major_formatter(ScalarFormatter())
        ax.set_xlabel('Full tensor/GPU (KiB; 1024 = 1 MiB)', fontsize=9)
        ax.set_ylabel('Latency (µs)')
        ax.grid(alpha=.25); ax.legend(fontsize=8, loc='upper left')
    fig.suptitle(title, fontsize=12)
    fig.text(.5, .01, 'FP32 · four NVIDIA L4 GPUs · median of three run means · all powers of two shown',
             ha='center', fontsize=9)
    fig.tight_layout(rect=(0, .035, 1, .94))
    fig.savefig(root/'docs/assets'/filename)
    fig.savefig(root/'build'/filename.replace('.svg', '.png'), dpi=120)

focus_plot([ar, ag, rs], 'collectives-4l4-1k-1m.svg',
           'UCCL Lite vs NCCL · ordinary calls · 1 KiB–1 MiB')
focus_plot([[r for r in resident if r['operation']==op]
            for op in ['all_reduce','all_gather','reduce_scatter']],
           'collectives-4l4-resident-1k-1m.svg',
           'Experimental prepared resident mode · 1 KiB–1 MiB · preparation excluded', True)
