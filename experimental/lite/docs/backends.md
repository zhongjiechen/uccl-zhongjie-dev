# Collective backend organization

Both NCCL libraries use the same frontend and profile parser. Each build links
one backend factory; it does not select a backend merely from the GPU count.
The existing exact topology/software/shape profile remains mandatory.

| Layer | Location | Responsibility |
| --- | --- | --- |
| NCCL API | `integration/nccl_lite/nccl_abi.cc` | Communicators, NCCL argument/error translation, profile admission, route accounting |
| Bootstrap/launcher | `integration/nccl_lite/host_control.h`, `backend_profile.py` | Rank consensus and verified runtime identity |
| Backend contract | `include/uccl_lite/backend.h` | Capabilities, construction, prepare, enqueue, completion and teardown |
| Shared runtime | `src/runtime/`, `include/uccl_lite/runtime/` | Backend-independent profile parsing and CUDA error utilities |
| Local backend | `src/backends/local/` | Eight-L4 BF16 host-packet transport, CPU workers and GPU kernels |
| Multinode backend | `src/backends/multinode/` | Two four-L4 FP32 hosts, local GPU channels, MSCCL++/RDMA and host-packet pipelines |

The frontend is compiled once into `build/runtime/nccl_abi.o` and linked into
both libraries without backend-specific preprocessor definitions. It obtains
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

`tests/nccl_mscclpp/strict_abi.cu` exercises both builds. Local tests and full
unchanged nccl-tests runs use `run_local8.py`; two-host runs use `run.py`.
`compare_local8_backends.py` freezes two libraries and compares the same checked
eight-rank per-call workload in reversed backend order. Each layout/size uses
two jobs, each with three repetitions of five warmups and ten timed calls;
the client checks every output and guard. It records library/profile hashes,
hardware, CPU affinity, raw samples and the complete 1-KiB–1-GiB comparison.
No best-per-size variant or sample trimming is used.

Historical qualification reports remain tied to their archived library and
source hashes. Refactor validation is recorded separately from those reports.
See the [eight-L4 refactor measurements and build checks](../benchmarks/data/backend_refactor_8l4_20261119/README.md).
