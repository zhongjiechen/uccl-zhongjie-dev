# Collective backend organization

All three NCCL libraries use the same frontend and profile parser. Each build links
one backend factory; it does not select a backend merely from the GPU count.
The existing exact topology/software/shape profile remains mandatory.

| Layer | Location | Responsibility |
| --- | --- | --- |
| NCCL API | `integration/nccl_lite/nccl_abi.cc` | Communicators, NCCL argument/error translation, profile admission, route accounting |
| Bootstrap/launcher | `integration/nccl_lite/host_control.h`, `backend_profile.py` | Rank consensus and verified runtime identity |
| Backend contract | `include/uccl_lite/backend.h` | Capabilities, construction, prepare, enqueue, completion and teardown |
| Shared runtime | `src/runtime/`, `include/uccl_lite/runtime/` | Backend-independent profile parsing and CUDA error utilities |
| Four-L4 backend | `src/backends/local4/` | Single-host FP32 CUDA IPC, packet, GPU ring and host-copy schedules |
| Eight-L4 backend | `src/backends/local/` | Eight-L4 BF16 host-packet transport, CPU workers and GPU kernels |
| Multinode backend | `src/backends/multinode/` | Two four-L4 FP32 hosts, local GPU channels, MSCCL++/RDMA and host-packet pipelines |

The frontend is compiled once into `build/runtime/nccl_abi.o` and linked into
all three libraries without backend-specific preprocessor definitions. It obtains
host count, ranks per host, dtype, element width, profile format and extent
limits from `backendCapabilities()`. Backend selection happens at link time:
each backend supplies `backendCapabilities()`, `createBackendId()` and
`createBackend()`.

`CollectiveBackend` is a NCCL/MSCCL++/MPI-independent C++ interface. Construction
is CPU-only; the first eligible call explicitly invokes `prepare()`. Calls are
serialized per communicator. `enqueue()` completes on the caller's CUDA stream
and each backend owns the staging, event ordering and transport retirement
needed to preserve buffer reuse. These algorithms and schedules are retained
from the qualified implementations, without adding a common staging copy or
an extra synchronization around every call.

## Backend contents

`src/backends/local/backend.cc` implements the contract and caller-buffer/
stream adaptation. `transport.cc` owns shared-memory preparation, NUMA placement
and CPU workers. `kernels.cu` contains all three packet collectives;
`packet_layout.cuh` is their private layout/launch interface. Its build flags
and dependency rules live in `module.mk`. README Testbeds 1 and 2 use this same
backend with their separately recorded NCCL controls.

`src/backends/multinode/backend.cc` implements the same contract around
`NetworkCollectives`. `collectives.cu` owns network collective scheduling and
MSCCL++ algorithm execution. GPU algorithms remain in `device_collectives.cuh`;
GPU/IB setup is in `algorithm_transport.h`. The host path uses
`shared_host_packets.h`, `host_packet_transport.h`, `async_host_gather.cuh` and
`persistent_host_gather.cuh`. `policy.mk` retains the qualified size policy and
`module.mk` owns its build rules. The local backend does not link MSCCL++/RDMA.

Standalone harnesses stay in `src/collectives/` and `benchmarks/`. Their existing
local C API and `NetworkCollectives` C++ API are declared in
`include/uccl_lite/backends/local.h` and `multinode.h`. The former
`multiprocess_collectives.h` and `collectives.h` headers are forwarding includes
for source compatibility, with no duplicate implementation.

## Build and verification

```bash
make nccl-local4      # build/nccl_local/libuccl_lite_nccl_local_strict.so
make nccl-local       # build/nccl_local8/libuccl_lite_nccl_local8.so
make nccl-multinode   # build/nccl_lite/libuccl_lite_nccl_strict.so
make backend-profile-check
```

The previous `nccl-local8-strict`, `nccl-mscclpp` and `nccl-mscclpp-strict`
targets, library filenames and loader aliases remain compatible. The log/profile
identifiers retain their existing spelling so archived audit tools can read
the results. `strict` still means no native NCCL fallback, not another backend.
The former `mscclpp_profile.py` launcher name is a symlink to
`backend_profile.py`, preserving older commands and launcher-content hashes.

`tests/nccl_mscclpp/strict_abi.cu` exercises all three builds. Unchanged
nccl-tests runs use `local.py` for four L4s, `run_local8.py` for eight L4s,
and `run.py` for two hosts.
`compare_local8_backends.py` freezes two libraries and compares the same checked
eight-rank per-call workload in reversed backend order. Each layout/size uses
two jobs, each with three repetitions of five warmups and ten timed calls;
the client checks every output and guard. It records library/profile hashes,
hardware, CPU affinity, raw samples and the complete 1-KiB–1-GiB comparison.
No best-per-size variant or sample trimming is used.

Historical qualification reports remain tied to their archived library and
source hashes. Refactor validation is recorded separately from those reports.
See the [eight-L4 refactor measurements and build checks](../benchmarks/data/backend_refactor_8l4_20261119/README.md).

## Four-L4 backend

`local4` adapts the retained four-process FP32 implementation to
`CollectiveBackend`; the shared NCCL frontend is unchanged. Its class and
operation names no longer masquerade as the network backend. Construction stays
CPU-only, and preparation, caller-stream completion, cached internal graphs and
transport retirement remain owned by the backend. `policy.mk` retains the
`four-l4-fp32` size policy. This is explicit profile admission, not automatic
routing based on GPU count. Positive-count application graph capture remains
unsupported; zero-count capture is accepted by the common frontend.

The `nccl-local-strict` target and `build/nccl_local/strict/libnccl.so.2` loader
alias are retained. The local profile magic is distinct from both other
backends. The launcher accepts one four-L4 host for FP32, two four-L4 hosts for
network FP32, or explicit `--local8` for eight-L4 BF16. CPU placement helpers
support the retained `core2` placement and fingerprint their own sources.

Run the checked stock-client sweep (external nccl-tests binaries are required):

```bash
make nccl-local4 build/nccl_local/strict_abi_check MPICXX=/path/to/mpicxx
python3 -B tests/nccl_mscclpp/local.py \
  --library build/nccl_local/libuccl_lite_nccl_local_strict.so \
  --real-nccl /path/to/native/libnccl.so.2 --tests /path/to/nccl-tests/build \
  --mpirun /path/to/mpirun --cpu-placement core2 \
  --max-bytes 1073741824 --unalign 1 --out build/local4-stock
```

Use a new output directory and add `--abi build/nccl_local/strict_abi_check`
for changing buffers/streams, delayed ranks, mixed layouts, guards, unsupported
calls and two communicator lifetimes. `backend-profile-check` covers all three
profile contracts. Historical four-L4 timing qualification does not transfer
to this refactored binary; new correctness evidence is recorded separately.

See the [four-L4 refactor validation record](../benchmarks/data/backend_refactor_4l4_20261120/README.md) for the completed single-node and two-node correctness results through 1 GiB.
