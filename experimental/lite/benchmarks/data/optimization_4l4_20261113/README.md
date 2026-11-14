# Four-L4 pipelined ReduceScatter, November 13, 2026

On `mibura-sky-test-02`, the new opt-in `--rs-ring` candidate overlaps peer
copy-engine transfers with GPU reduction of 8 MiB pieces on a separate stream.
Three ring steps use order 0→1→3→2→0, immutable receive slots and per-tile CUDA
events. Each public stream joins its reduction stream before completion;
all-rank entry events protect scratch across asynchronous calls. FP32 inputs
remain immutable during each call and the result is out of place.

The full sweep's 1 GiB result is **39.446 ms**, versus the original peer path
**48.595 ms** and paired NCCL **48.972 ms**: **1.232×** and **1.242×** speedups.
Bus bandwidth is **20.416 GB/s**, using 0.75 × full input bytes / seconds.
The first shorter sweep measured 39.436 / 48.627 / 47.495 ms; NCCL varies across
windows, so use each run's own paired control. Do not compare against an older
NCCL window or claim an identical NCCL ratio across runs.

Hardware: four NVIDIA L4s, PCIe Gen4 x16, NUMA pairs (0,1)/(2,3), dual Xeon
Silver 4410Y, CUDA toolkit/runtime 12.9, driver 595.58.03, NCCL 2.28.7 default.
The full run has 10 warmups and three rotated rounds of 10 timed calls per
candidate/size. Maximum-GPU per-call CUDA events include host enqueue gaps;
allocation, fill and verification are outside timing. Every warmup and timed
output is checked. `selected.csv` and the SVG choose the fastest measured UCCL
candidate at each size, independently of NCCL. This is a benchmark envelope,
not a portable runtime routing policy. The ring remains explicitly opt-in.

| Full input/GPU | Selected ring | Original peer | Paired NCCL |
| ---: | ---: | ---: | ---: |
| 16 MiB | 725.54 µs | 799.33 µs | 905.55 µs |
| 64 MiB | 2.554 ms | 3.057 ms | 3.238 ms |
| 256 MiB | 9.939 ms | 12.239 ms | 12.596 ms |
| 1 GiB | 39.446 ms | 48.595 ms | 48.972 ms |

Reproduce from `experimental/lite`:

```bash
make peer_ag_rs peer-ring-check
./build/uccl_lite_peer_ring_check
./build/uccl_lite_peer_ag_rs --rs-only --rs-ring \
  -b 1024 -e 1073741824 -w 10 -n 10 -r 3
python3 scripts/summarize_rs_pipeline.py
```

The summarizer consumes `rs_full.log`, validates 2,940 timed samples and writes
all 21 selected rows plus the README figure. `rs_final_1g.log` is a separate
final-build confirmation (5 warmups, 3×10 calls): ring 39.664 ms, peer
48.676 ms, NCCL 46.425 ms, or 1.227× over peer and 1.170× over NCCL. The full sweep preceded only
clang-format-14 changes; the final build/check revalidates the formatted source.
`identity.json` records final source/binary/library hashes, not a purported hash
of earlier experimental binaries.

The additional independent check uses signed pseudo-random rank/position/
generation-dependent values and a double-precision host sum, with absolute plus
relative tolerance 1e-6. It tests 1008, 262160, 8388624 and 67108880 bytes, rotates
a delayed rank and enqueues four changed generations without host completion,
checking every fourth result (12 generations per size). The standard benchmark
checks every invocation. Both tests are eager; graph capture is not claimed.

Negative controls remain: whole-shard hierarchy `--rs-hier` reached 53.727 ms
at 1 GiB; the unpipelined ring reached 50.000 ms. Cutting socket traffic or
changing to a ring without copy/reduction overlap did not improve that size.
The original and unpipelined source snapshots and all raw logs are retained.
The hierarchy is experimental and is not included in the published envelope.

No CPU-router core or selection policy changed. The completed AR, AG, geometry and graph experiments are recorded in
[EXPERIMENTS.md](EXPERIMENTS.md), including the faster experimental hybrid AR
and the negative tuning results. `PROGRESS.md` records the execution outcome.
