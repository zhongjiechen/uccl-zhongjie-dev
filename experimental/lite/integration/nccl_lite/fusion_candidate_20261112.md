# SGLang EP8 fusion boundary (2026-11-12)

This is a runtime inventory for the GLM-4.5-Air-FP8, TP8/DP8/EP8 setup in
`launch_glm45_air.sh`. It is not a fused implementation or a performance result.

## Actual producer and consumer

In the installed SGLang build, `Glm4MoeDecoderLayer.forward` calls
`LayerCommunicator.prepare_attn`, attention, `prepare_mlp`, MoE/MLP, and
`postprocess_layer` in order. With `--enable-dp-attention`,
`--moe-dense-tp-size 1`, and `--moe-a2a-backend none`, AG can occur in
`CommunicateWithAllReduceAndLayerNormFn._gather_hidden_states_and_residual`
through `dp_gather_partial` before MoE/MLP. RS can occur in
`CommunicateSummableTensorPairFn._scatter_hidden_states` through
`dp_reduce_scatter_tensor` after MoE/MLP. Other AGs can occur at model input
and in attention layout transitions. A NCCL symbol interposer does not know
which GEMM consumes or produced its tensor.

The model's quantization config uses FP8 weights with dynamic per-token FP8
activations. The SGLang runtime selects `CompressedTensorsW8A8Fp8MoE` for
the routed experts. The existing eight-L4 `router_fusion_8l4.cu` uses BF16
cuBLAS/CUTLASS GEMM and a single-process `PacketPipeline`; neither its GEMM
arithmetic nor its process ownership matches this SGLang MoE path. Porting
the benchmark call directly to the NCCL compatibility layer would give a
different operation.

The MoE gate itself is a BF16 linear projection with `K=4096` and `N=128`
routed experts (the checkpoint's gate weight is `[128,4096]` BF16). A fresh
standalone eight-L4 AG+GEMM probe on this host used the
`router_fusion_8l4` binary (CUDA 12.9, NCCL 2.28.9), one warmup round and
two measured rounds of five calls per method. Its slowest-GPU CUDA-event
medians from the [30 samples per shape](fusion_ag_gate_probe_20261112.tsv)
were:

| Total gathered rows | cuBLAS + NCCL | cuBLAS + router | fused + router |
| ---: | ---: | ---: | ---: |
| 8 (one row/rank decode) | 0.2519 ms | 0.1904 ms | **0.1331 ms** |
| 296 (37 rows/rank prefill) | 0.6252 ms | **0.5371 ms** | 0.6221 ms |

The 296-row shape matches the observed AG count of 151552 BF16 elements per
rank (`37 × 4096`). The 8-row case is a candidate for a multi-process
AG+gate GEMM; the prefill result is a negative control. These are isolated
single-process operations with synthetic BF16 weights, not SGLang timings.

## Shape probe

One eight-request, 64-input, 16-output-token `/generate` batch was run with
`--skip-server-warmup`, packet mode, and
`UCCL_LITE_NCCL_TRACE_SHAPES=1`. Rank 0 first observed these BF16 NCCL
element counts:

| Operation | Counts first observed |
| --- | --- |
| AG | 4096, 18944, 132608, 151552 |
| RS | 4096 |
| AR | 4096, 28672, 32768, 262144, 1839104 |

These are first-occurrence shapes, not call frequencies. The server kills
workers on shutdown, so its C++ destructor did not print the final histogram.
The cold request took 4.85 s and is not a comparison with NCCL. The trace is
under `build/fusion_shape_server.log` on the measured host; it is generated
output and is not committed.

## Further integration requirements

Fusion must enter at a SGLang layer boundary that has the weights and token
layout, not only at `ncclAllGather` or `ncclReduceScatter`. The
[MoE epilogue plus RS prototype](fusion_rs_8l4_20261112.md) uses such a
boundary for the 4096-element RS shard. The subsequent
[AG+gate integration](fusion_ag_rs_8l4_20261112.md) adds an eight-process,
stream-ordered path for the one-row decode shape. Its cuBLAS option preserves
the measured gate BF16 logits; the cooperative GEMM option changes some of
them and remains experimental. Dynamic FP8 expert quantization and routing
stay in SGLang. Unsupported shapes and layouts fall back to the existing
packet/NCCL path. The current report retains both gains and regressions.
