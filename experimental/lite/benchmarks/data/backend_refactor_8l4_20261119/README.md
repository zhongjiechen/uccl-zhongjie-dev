# Shared-backend refactor: eight-L4 validation

This measures the refactored NCCL frontend/runtime and local backend against
fresh runs of the preserved, qualified pre-refactor `.so` from `drop-in` at
`5c0be574`. The [backend organization](../../../docs/backends.md) describes the
shared interface and the separate local/multinode implementations. The CUDA
algorithms and tuned schedules are unchanged; all 11 local kernels have
identical SASS after normalizing filename-derived anonymous-namespace names.

## Performance

Positive changes mean the refactor is slower. These aggregate ratios cover
all 21 power-of-two sizes from 1 KiB through 1 GiB and both layouts.

| Collective | Median latency change | Geometric-mean change | Worst point | Points >5% slower |
| --- | ---: | ---: | ---: | ---: |
| AllReduce | +0.25% | +0.71% | +7.32% | 2/42 |
| AllGather | -0.11% | -0.99% | +1.08% | 0/42 |
| ReduceScatter | -0.02% | -0.11% | +3.76% | 0/42 |

The two points above 5% are 1-KiB AllReduce: out of place is 20.992 → 22.528 µs
(+1.536 µs); in place is 19.456 → 20.480 µs (+1.024 µs). The independent
out-of-place jobs vary (baseline 19.456/22.528 µs, refactor 25.088/19.968 µs).
These results are retained. The campaign does not establish exact zero
slowdown at every point or statistical significance for each small difference.

At 1 GiB, latency in milliseconds (AR in place; AG/RS out of place):

| Collective | Pre-refactor `.so` | Refactored `.so` |
| --- | ---: | ---: |
| AllReduce | 261.714 | 261.634 |
| AllGather | 154.556 | 154.559 |
| ReduceScatter | 167.537 | 166.565 |

[All 126 points](comparison.csv), [summary](summary.json), and
[separate native NCCL comparison](native_comparison.json) are retained.
The native comparison uses nccl-tests batch timing and is not mixed with
these per-call ratios; native NCCL is faster at 23/126 points in that campaign.

## Configuration and protocol

- `wxz-l4`: eight NVIDIA L4s, four per NUMA node; Intel Xeon 2.20 GHz,
  48 reported cores/96 logical CPUs, driver 580.178.04, CUDA 12.9, sm_89,
  NCCL 2.28.9 headers/native control, Open MPI 4.1.2.
- BF16 AG and SUM AR/RS. Bytes are complete AG output or AR/RS input per rank.
  Both libraries use eight MPI processes, one GPU each, the same CUDA client,
  packet configuration, CPU worker placement and host environment.
- Two independent jobs per backend/layout/collective, reversing backend and
  layout order in the second job. Each size/job has three repetitions of five
  warmups and ten checked timed calls: **15,120 timed calls and 7,560 warmups**.
- Input-ready maximum-rank per-call CUDA events include host submission, ABI
  staging/copies and stream completion. Preparation, input generation and
  verification are excluded. Every warmup and timed call checks the full
  output and guard. Points aggregate repetition medians, then job medians.
- One fixed library per backend, no best-per-size selection or trimmed samples.
  Heavy dependency/core builds started after timing finished. Process-tree
  guards reject unrelated GPU work; every full job audits worker/core affinity.
- Baseline SHA-256: `a613c4f85541aaf39c5e08b4db2d12ee817b29942cbd0cb8dde423df0cb854c7`.
  Refactor SHA-256: `ce862a6f0df10437c25c09eaf5859dd97c71c8cb81a42bfc93c3a0b144abe02b`.

[Hardware](hardware.json), [provenance](provenance.json),
[source hashes](source_hashes.json), [validation](validation.json) and
[independent raw-log audit](audit.txt) identify the exact measured build.

## Validation and limits

The refactored local library passed twelve complete unchanged nccl-tests
sweeps, 234 ABI calls per rank across two communicator lifetimes, 128 queued
mixed calls per rank, explicit unsupported-call/profile rejection and three
direct loader-alias sweeps. ABI exports match the baseline; the local library
has no native NCCL or MSCCL++ dependency. The shared profile parser passed
57 checks. Existing router policy (35), dispatcher (74), CUDA dispatcher,
four-device (0/1/4/5) and eight-GPU eager/graph checks passed; graph tests cover
1,000 mixed collectives and five replays. These core graph tests do not imply
application-capture support in the NCCL adapter.

The complete multinode library, its size/queue clients and standalone harness
compiled against upstream MSCCL++ revision `52f659dac390c9820bdc4db0577faeffe61a7370`.
The historical MSCCL++ installation included a separately recorded local patch;
this fresh dependency build is compilation evidence. **No two-host runtime or
performance claim is made for the refactor:** the second four-L4 host was
unreachable. Both older integration libraries also compiled after source moves.

## Reproduction

Build into a separate directory to preserve the qualified baseline:

```bash
make -j4 nccl-local backend-profile-check \
  build/refactor/nccl_local8/strict_abi_check \
  build/refactor/nccl_local8/reference_timing_check \
  build/refactor/nccl_local8/queue_check \
  BUILD=build/refactor CUDA_HOME=/usr/local/cuda-12.9 NCCL_HOME=build/nccl-2.28.9

python3 -B tests/nccl_mscclpp/run_local8.py \
  --library build/refactor/nccl_local8/libuccl_lite_nccl_local8.so \
  --abi build/refactor/nccl_local8/strict_abi_check \
  --queue build/refactor/nccl_local8/queue_check --checks \
  --out build/my-refactor-nccl-tests

python3 -B tests/nccl_mscclpp/compare_local8_backends.py \
  --baseline /path/to/preserved/libuccl_lite_nccl_local8.so \
  --candidate build/refactor/nccl_local8/libuccl_lite_nccl_local8.so \
  --client build/refactor/nccl_local8/reference_timing_check \
  --out build/my-refactor-paired
```

[measurements.tar.gz](measurements.tar.gz) retains raw logs, manifests, profiles,
launchers, measured sources, build logs and checks, excluding compiled binaries
and vendored dependency trees. [raw_checksums.json](raw_checksums.json) authenticates
archive members. Frozen executables remain under `build/backend_refactor_20261119/`.
Historical reports and README measurement curves are unchanged.
