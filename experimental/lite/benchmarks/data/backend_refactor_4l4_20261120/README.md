# Four-L4 backend refactor validation, 2026-11-20

Worktree: `refactor/collective-backends`, based on `de07cb7d`.
The four-L4 FP32 drop-in algorithms are now a third backend linked to the
same NCCL frontend/profile objects as the eight-L4 and network libraries.
`local4-source-audit/equivalence.json` verifies the port preserves the algorithm
body after class/type/include renames and clang-format 14 formatting.
Source hashes and raw validation records are included here.

Hardware: mibura-sky-test-02 and mibura-sky-test-01, four NVIDIA L4s each,
10.10.55.2 / 10.10.55.1, interface ibp55s0f0. Other GPU workloads were active during the early screens; both hosts were
free of those GPU processes before the final 1-GiB campaigns.
This record establishes correctness only; timings under contention do not
qualify performance or replace historical fixed-library qualification.

Completed checks:

- All three shared-library targets and the three corresponding ABI clients build.
- 80 shared profile checks (including cross-backend rejection); 14 CPU-placement
  checks; clang-format 14 and `git diff --check` pass.
- Single-node stock nccl-tests: AR/AG/RS, FP32 SUM, 1 KiB–1 GiB, both layouts,
  one-float misalignment, 126 size/layout cases, numerical errors zero.
- Single-node ABI: 198 calls/rank, two lifetimes, full output/guards, changing
  pointers/streams, mixed layouts, delayed ranks, unsupported calls.
- Two-node stock nccl-tests: AR/AG/RS, FP32 SUM, both layouts,
  aligned and one-float misalignment through 1 GiB; all correctness and route assertions pass.
- Two-node graph/product/missing-profile/group rejection checks pass.
- Two-node ABI: 198 calls/rank, two lifetimes, all full-buffer checks pass.
- Candidate processes report native NCCL absent and zero fallback. The local4
  library has SONAME libnccl.so.2 with no native NCCL dependency.

Single-node stock timing uses five warmups/ten iterations, maximum-rank batch
completion; network uses five warmups/twenty iterations and one native/candidate/
candidate/native round, plus a misaligned candidate sweep. Baseline is native
NCCL 2.28.7. Each candidate is one fixed library; no best-per-size selection.
These are correctness screens, not a performance acceptance comparison.

Negative result: the single-node 1-GiB stock sweep failed during preparation
with CUDA out-of-memory while unrelated jobs occupied about 19 GiB per GPU.
The failure is retained in `validation_local4_stock`; it is not a pass.
The memory constraint subsequently cleared. Fresh full 1-GiB local and
two-node campaigns both pass on the same recorded libraries, including
misalignment and exact route assertions; the two-node rejection and ABI checks
also pass again. Their records are `validation_local4_1g` and
`validation_network_1g` in the archive. The earlier OOM is retained as a
resource failure, not a correctness pass.
The 256-MiB local and network campaigns have completed successfully, as has
ABI confirmation against the final formatted local build. The local ABI audit
also checks exact per-operation/size/rank counts in each lifetime (198 calls),
and rejects corrupted call counts, incomplete completion markers and fallback.
The local 512-MiB sweep also passes all 120 size/layout cases. No eight-L4
runtime result is claimed by this change.

`validation.tar.gz` contains raw logs, commands, identities, routes and build
checks. `sources.tar.gz` contains project dependencies read from compiler depfiles
plus Make policies, launchers and validation drivers; `source_hashes.json` lists
their hashes. `checksums.json` covers the archives and report. Library and shared
frontend object hashes are in `local4-source-audit/binaries.json` in the archive.
