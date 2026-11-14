# Execution outcome

Completed the requested full-range pass, with small/medium optimization and
publication of complete per-size data rather than only 1-GiB headline numbers.

Accepted implementations and evidence:

- Native AR: input-ready stream boundary repaired; changing-generation and
  nonfinite checks; explicit 1-/8-CTA variants; 21-size matched final sweep.
  The 1-MiB original kernel passed 200 timed changed-input stress calls after
  fixing input readiness. Fence-only experiments did not fix the defect.
- AG: eager fused host staging closes the 4-MiB loss and improves 1–16 MiB;
  27-size paired sweep, source-matched final idle-window recheck, independent
  delayed/async/tail checks and isolated sanitizer pass.
- RS: phased LL hierarchy improves 16–32 KiB; the previous fused and ring
  paths remain in the full 27-size comparison; independent/sanitizer checks.
- Resident mode: prepared single-CTA workers, packed 64-bit descriptor,
  LL only at small validated cutoffs, fused RS+AG for larger AR. Full 17-size
  AR/AG/RS sweep, 83-operation independent host-reference test and sanitizer
  pass, plus separate eager/global-graph/per-device-graph NCCL controls.
- Complete ordinary tables, a separate resident table, small/medium/large
  figures, command recipes, source snapshots, negative controls and identities.

Excluded from accepted results:

- Native AR runs before the input-ready correction (`ar_full/` and early
  geometry/LL screens); the summarizer rejected a bad 1-MiB SM result.
- The resident prototype with mismatched LL bounds, even though its numerical
  checker passed. Shared constants and independent boundary/memcheck tests
  establish the accepted allocation/dispatch contract.
- The NCCL-initializing sanitizer run (tool/library initialization errors and
  timeout); the isolated candidate checks passed with zero memory errors.
- The graph-capable mini-host timing probe at 21:03 UTC, when another workload
  used the GPUs. Its source is archived; the validated eager implementation
  remains active. GPU process monitoring confirmed the final medium AG recheck
  and final independent check had no observed foreign process.

Boundaries:

These are standalone opt-in paths. Resident mode changes the execution contract
and reserves GPU resources; it is not a stream-enqueue replacement or an
end-to-end application speedup. No automatic GPU-count-based dispatcher policy
or application hook was introduced. Ordinary native AR still has nominal
32-KiB and 4-MiB losses, and all min/max ranges are available. No claim is made
that every possible future optimization has been exhausted.
