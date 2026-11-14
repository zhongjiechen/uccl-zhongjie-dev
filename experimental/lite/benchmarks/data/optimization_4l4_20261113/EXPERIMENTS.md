# Additional optimization experiments

All measurements below use the same four-L4 owner host described in README.md.
No rejected tuning is promoted just because it reduces traffic or block count.
Raw logs are in `tuning/`; all original controls remain available.

## ReduceScatter granularity

The validated default is 8 MiB per ring piece. A subsequent 1 GiB sweep used
3 warmups and 3 rotated rounds of 5 checked calls, FP32 out of place,
maximum-GPU CUDA-event timing. Ring times:

| Piece | Time |
| --- | ---: |
| 1 MiB | 40.891 ms |
| 4 MiB | 39.746 ms |
| 8 MiB (previous accepted window) | 39.446 ms |
| 16 MiB | 39.888 ms |
| 32 MiB | 44.920 ms |

4/8/16 MiB are close; 8 MiB remains the established choice. There is no
per-size retuning in the published ring candidate. The small differences
across windows do not prove a universal optimum.

## AllGather packet geometry

Six candidates varied the packet cap and transport-block cap with unchanged
per-source GPU progress. Each 1 GiB window paired packet AG, direct peer and
NCCL with 3 warmups and 3×5 checked calls. FP32 bits are copied without dtype
conversion; results are full output/GPU and maximum-GPU CUDA-event latency.

| Block cap | Packet cap | Packet AG |
| ---: | ---: | ---: |
| 64 | 1 MiB | 39.635 ms |
| 64 | 8 MiB | 38.959 ms |
| 128 | 1 MiB | 38.069 ms |
| 128 | 8 MiB | 37.895 ms |
| 248 | 1 MiB | 38.527 ms |
| 248 | 8 MiB | 37.403 ms |

The established 248-block/4-MiB path measured 36.429 ms in the initial
confirmation window. None of these candidates justifies changing that
geometry. A fresh same-geometry final sweep is required for README publication.
The harness now exposes `--packet-bytes` and `--packet-blocks`; these do not
modify the CPU-router core or its calibration policies.

## Graph measurement contract

The initial attempt to capture the direct peer-copy path failed at
`cudaMemcpyPeerAsync` on CUDA 12.9 (see `graph_diagnostic.log`). Supported graph
experiments are restricted to SM pull/push-serial candidates and NCCL through
1 MiB. LL/fused flags that embed a host generation are excluded: replaying a
captured constant generation would not be a correct protocol.

One fork/join graph spans the four GPUs. Capture/instantiation and changed-input
fill are excluded from timing. Each replay records per-GPU external timing
events and joins all GPU ends to the origin stream before validation. Every
warmup and timed replay checks complete output. It does not divide a graph
candidate by eager NCCL. The earlier capture with ordinary internal event nodes
failed `cudaEventElapsedTime`; the corrected graph uses external record nodes
for actual replay timestamps. Failed captures are diagnostics, not timings.

The completed supported graph sweep (`graph_full.log`) has 10 warmups and
3×10 checked calls, 11 sizes (1 KiB–1 MiB), two collectives: **1,680 timed
samples**. The 1008-byte scalar tail also passes. Paired graph results:

| Operation | Full bytes/GPU | Selected SM candidate | UCCL / NCCL (µs) |
| --- | ---: | --- | ---: |
| AG | 1 KiB | pull | 12.218 / 20.502 |
| AG | 32 KiB | push-serial | 13.088 / 23.350 |
| AG | 1 MiB | push-serial | 69.178 / 93.690 |
| RS | 1 KiB | pull | 11.837 / 20.512 |
| RS | 32 KiB | push-serial | 37.712 / 23.600 |
| RS | 1 MiB | push-serial | 83.318 / 84.112 |

This selects per-size UCCL variants within the supported graph set, not against
NCCL. In particular, graph RS at 32 KiB loses and is worse than the earlier
eager fused-fence candidate. It is not a reason to default all callers to graph
mode. The protocol events are re-recorded after graph destruction before any
subsequent eager calls: internal captured event dependencies are not live eager
event records. Failed lifecycle/capture diagnostics are retained separately.

Reproduce: `./build/uccl_lite_peer_ag_rs --graph -b 1024 -e 1048576 -w 10 -n 10 -r 3`.
The ordinary eager path and its selected envelopes remain separate.

## AllReduce launch geometry and ring granularity

The native AR harness adds only a diagnostic `-a INDEX` algorithm filter; index
8 selects the existing A3 ring. No publication fence was weakened. Six 1 GiB
configurations used FP32 in-place, four processes, slowest-rank host call timing,
2 warmups and 5 checked calls each. There is no paired NCCL in these tuning
windows and no per-size best-geometry curve.

| Blocks × threads | Piece target | Mean latency |
| --- | ---: | ---: |
| 8 × 512 | 2 MiB | 80.132 ms |
| 8 × 512 | 8 MiB | 80.114 ms |
| 24 × 512 | 2 MiB | 80.292 ms |
| 24 × 512 | 8 MiB | 80.230 ms |
| 48 × 512 | 2 MiB | 80.956 ms |
| 48 × 512 | 8 MiB | 80.962 ms |

The code caps pieces at 16 per quarter, so at 1 GiB both requested piece targets
produce 16 pieces and an actual 16 MiB extent. This is a geometry/cap check,
not evidence that actual 2 MiB and 8 MiB transfers perform identically.
The ~0.1% difference between 8 and 24 blocks is insufficient to promote a
new default from these single windows; 48 blocks regresses. The original
24×512 and 8 MiB requested target remain unchanged.

At 64 MiB, where the piece targets are not capped to the same value, 2 MiB
pieces took 5.569 ms versus 5.253 ms for 8 MiB (3 warmups, 10 checked calls).
This also supports retaining the established target.

The final native 1 GiB AR confirmation alternates three UCCL/NCCL windows,
5 warmups and 10 timed calls in each, four processes, FP32 in-place,
slowest-rank host enqueue+completion timing. Median run means are **80.243 ms**
for A3 ring and **90.364 ms** for NCCL. All checked errors are zero.
`ar_uccl_paired_*.log` and `ar_nccl_paired_*.log` contain 60 timed samples.
The earlier attempted NCCL command with 3 warmups/5 calls was rejected by the
baseline's minimum-count validation; it supplied no measurement.

## Experimental hybrid AllReduce

`make peer-hybrid-allreduce` builds `uccl_lite_peer_hybrid_allreduce`, an explicit
one-process FP32 in-place candidate. It composes the new ring ReduceScatter
with the existing per-source mapped-host AllGather. RS writes a final immutable
shard to temporary output; AG reads that shard and overwrites the original
full input with the reduced result. No extra full-array output copy is needed.
Per-stream completion and packet ownership protect the two stages.

`hybrid_final.log`: 16 MiB–1 GiB, fixed 8 MiB RS pieces and 4 MiB AG packets,
248-block AG cap, 10 warmups and three alternating rounds of 10 calls per
backend. Every warmup and timed call checks all elements on all GPUs. The timer
is maximum-GPU eager CUDA-event duration, including submission gaps. The paired
NCCL 2.28.7 control is also one-process FP32 in-place with the identical timer.
This curve uses one fixed candidate, **no per-size variant selection**.

| Full input/GPU | Hybrid AR | Paired NCCL |
| ---: | ---: | ---: |
| 16 MiB | 1.376 ms | 1.485 ms |
| 64 MiB | 4.939 ms | 5.514 ms |
| 256 MiB | 19.179 ms | 21.753 ms |
| 1 GiB | **75.978 ms** | **91.165 ms** |

At 1 GiB the matched NCCL speedup is **1.200×**. The first shorter sweep
measured 76.111 versus 89.869 ms. The candidate is experimental and opt-in;
it is not a replacement for the multiprocess native AR API. The native AR
80.243 ms is a different process/timing contract and is not the denominator
of a claimed hybrid speedup. At a 1008-byte tail, hybrid takes 127.3 µs versus
NCCL 32.4 µs, a retained negative result. Do not select it solely by GPU count.

`make peer-hybrid-check` builds an independent signed-input host-reference
check for the composition. It passes 1008, 262160, 8388624 and 67108880 bytes,
12 changing generations per size, rotating delayed producers, with groups of
four calls enqueued asynchronously and every fourth output fully checked.
Both stage-specific RS checks and the composition check use a double host sum
and absolute+relative tolerance 1e-6. Hybrid graph capture is not supported or
claimed. The existing router graph regression is separate.

Reproduce the fixed large-message curve:

```bash
make peer-hybrid-allreduce peer-hybrid-check
./build/uccl_lite_peer_hybrid_check
./build/uccl_lite_peer_hybrid_allreduce -b 16777216 -e 1073741824 \
  -w 10 -n 10 -r 3
python3 scripts/summarize_latest_4l4.py
```

The summarizer checks all **420 timed samples**, writes `ar_hybrid_summary.csv`
and the hybrid SVG. Final regression logs also retain backend policy,
dispatcher, four-GPU eager GPU-progress reduction (26,569,136 scalar/guard
checks), and four-GPU per-source graph checks (1,000 captured mixed collectives,
five replays, producer/consumer dependencies and ring wrap). The CPU-router
core and dispatch policy source were unchanged.

## Final AllGather confirmation

`ag_final_full.log` confirms the unchanged default per-source packet path:
**36.463 ms** at 1 GiB versus paired peer **45.507 ms** and NCCL **48.656 ms**.
It contains **2,070 checked timed calls**, 21 sizes, 10 warmups and 3×10 calls,
with the same FP32 out-of-place event timing as the earlier geometry windows.
`ag_selected.csv` and the refreshed SVG select the lowest UCCL median per size
without consulting NCCL. The earlier packet sweep and figure remain archived.

The final binary's independent 1 GiB hybrid recheck (5 warmups, 3×10 calls)
measured **75.883 ms** versus NCCL **86.570 ms** (**1.141×**). This confirms
candidate timing while exposing baseline variability; do not report the
full-sweep 1.200× ratio as invariant. The final source also explicitly rejects
`--graph` for hybrid AR rather than silently ignoring that option.

## Small native AllReduce geometry/proxy trial

The final tuning trial retained 2,400 fully checked timed calls: hierarchy LL
and tree LL at 1–16 KiB, four block/thread/proxy configurations, three windows,
10 warmups and 20 calls per size. It uses the native four-process FP32 in-place
host timer, without a paired NCCL run. `tuning/ar_small_summary.csv` preserves
all median/min/max run means; the raw logs include every call and zero errors.

Tree LL median examples (µs):

| Configuration | 1 KiB | 16 KiB |
| --- | ---: | ---: |
| Default 24×1024, proxy enabled | 15.802 | 24.641 |
| 1×256, proxy enabled | 18.064 | 38.043 |
| 1×256, proxy disabled | 17.462 | 37.691 |
| 4×256, proxy disabled | 13.648 | 27.772 |

The tiny-message candidate trades a lower 1-KiB median for a 16-KiB regression;
window variation is substantial, and proxy enablement and block geometry both
change in the last row. This is not sufficient evidence for a new global
default or a new advertised speedup against NCCL. The existing native defaults
remain intact, and no small-AR timing from this trial is spliced into another
host's curve.
