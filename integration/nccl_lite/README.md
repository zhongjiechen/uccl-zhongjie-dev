# UCCL Lite NCCL AG/RS/AR compatibility layer

This is an experimental **single-node, eight-process, BF16** NCCL compatibility
layer for AllGather, ReduceScatter(SUM), and AllReduce(SUM). It forwards other
NCCL calls used by the installed SGLang PyNccl wrapper to the real NCCL
library. It is not a complete NCCL ABI implementation.

The UCCL Lite data path lives in
`../../experimental/lite/src/collectives/multiprocess_collectives.cc`. Each
process owns one GPU and registers only its own portion of a POSIX shared-memory
buffer as CUDA host memory. A call queues GPU-to-host copy, a host callback that
routes or reduces BF16 data, and host-to-GPU copy on the caller's CUDA stream.
For AR, ranks reduce disjoint output ranges, synchronize, and copy the ranges
into their own result buffers.
The host callback never calls CUDA. A shared sequence and rank barrier keep
successive collectives from reusing staging memory early. The proxy maps NCCL
communicators by unique ID and rank; NCCL still handles initialization and all
other operations.

## Build and checks

```bash
make -C integration/nccl_lite
make -C integration/nccl_lite check
make -C integration/nccl_lite check-torch
```

Set `CUDA_HOME` and `NCCL_HOME` to the CUDA and NCCL header installations if
they differ from `/usr/local/cuda` and `/usr`. Set `REAL_NCCL` to an absolute
path when `libnccl.so.2` is not on the dynamic loader search path. Set
`TORCHRUN` if PyTorch's launcher is not on `PATH`.
`check` runs an eight-process C++ AG/RS/AR correctness test. `check-torch` verifies
that `LD_PRELOAD` intercepts PyTorch's NCCL ProcessGroup calls, including the
`ncclCommInitRankConfig` path. Both require eight idle GPUs. Generated files
stay under `integration/nccl_lite/build/`.

## Use with SGLang

Set both variables before launching the server and its worker processes:

```bash
export LD_PRELOAD="$PWD/integration/nccl_lite/build/libuccl_lite_nccl_compat.so"
export SGLANG_NCCL_SO_PATH="$LD_PRELOAD"
export UCCL_LITE_NCCL_ENABLE=1
export UCCL_LITE_NCCL_MAX_BYTES=33554432
export UCCL_LITE_REAL_NCCL=/path/to/the/real/libnccl.so.2
```

`LD_PRELOAD` reaches PyTorch ProcessGroup NCCL calls. `SGLANG_NCCL_SO_PATH`
reaches SGLang's ctypes based PyNccl wrapper. The proxy logs the first
intercepted AG, RS, and AR per communicator and rank. If these messages are absent,
the benchmark is measuring NCCL. Set `UCCL_LITE_NCCL_ENABLE=0` to pass through
all operations while keeping the same library loaded.

For request-level verification, set `UCCL_LITE_NCCL_TRACE=1` before launch.
It logs every successfully enqueued AG, RS, and AR call for rank 0, including
per-communicator cumulative counters. Record the counters after server warmup,
send one request batch at a time, and compare the counters after each response.
Tracing adds logging overhead, so disable it for latency measurements.

On Linux, `UCCL_LITE_MP_CALLBACK_CPUS=5,11,17,23,29,35,41,47` optionally
maps CUDA device indices 0 through 7 to one CPU each for the host callback.
It is unset by default. The callback changes only its own thread's affinity;
application scheduler threads need a separate affinity setting. On the tested
eight-L4 host, SGLang's `SGLANG_SET_CPU_AFFINITY=1` improved this proxy's
large-batch latency, while reserving one physical core per callback made it
slower. Keep callback pinning opt-in and measure it with the target workload.

In the validated eager SGLang setup, DP attention padding depends on the
active batch. Request-level tracing confirmed AG/RS during both prefill and
decode at batch size 8; smaller batches can take a different path. Check
request-level counters for every workload before attributing latency to this
proxy.

## Supported call boundary

- Eight ranks, one GPU per process, one fixed CUDA stream per communicator.
- BF16, equal per-rank counts, out-of-place AG and RS SUM, and in-place or
  out-of-place AR SUM. Maximum complete input/output bytes per rank are
  controlled by `UCCL_LITE_NCCL_MAX_BYTES`.
- Calls within `ncclGroupStart`/`ncclGroupEnd`, other datatypes, operations,
  rank counts, and oversized messages pass to NCCL **before enqueue**.
- CUDA Graph capture, variable-size gather/scatter, multi-node routing, and
  generic concurrent streams are outside this boundary.
- Numerical RS and AR output is BF16 round-to-nearest-even after FP32 accumulation.
  It can differ by one or more BF16 ulps from a different NCCL reduction tree.

The shared-memory rendezvous times out after 60 seconds and aborts all work on
a failed rank rather than publishing stale output. The last successful rank
to join unlinks the shared-memory name immediately; active mappings remain
valid until the processes exit. A crash before all ranks join can leave an
orphan under `/dev/shm/uccl-lite-mp-*`.
This compatibility layer has no automatic performance selection. Benchmark
results should identify which calls were intercepted and retain NCCL controls.
