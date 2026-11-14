# Four-L4 optimization execution outcome

The prioritized execution pass is complete: implement the highest-value RS
pipeline, evaluate AR and AG schedules/geometry, measure supported graph
latency with a matched baseline, run correctness/regression gates, and publish
current measurements in the lite README. This does not claim that every
possible hardware-specific collective technique has been exhausted.

Implemented and retained:

- Opt-in tiled ring RS: copy-engine/reduction overlap, immutable receive slots,
  per-tile events and asynchronous invocation boundaries. 1 GiB 39.446 ms
  versus old peer 48.595 ms; independent confirmation 39.664 ms.
- Experimental in-place ring-RS + host-packet-AG: 1 GiB 75.978 ms versus paired
  NCCL 91.165 ms; independent final build 75.883 versus 86.570 ms.
- Explicit peer/NCCL graph measurement mode for supported SM paths ≤1 MiB;
  host-generation LL/fence paths and unsupported CE capture are excluded.
- Diagnostic AR algorithm filter and explicit AG/RS geometry knobs, with
  original defaults retained where tuning did not justify changes.

Measured and rejected/retained separately:

- Whole-shard hierarchy and unpipelined RS ring lose at 1 GiB.
- RS piece-size sweep supports retaining 8 MiB.
- Six AG packet/grid variants do not improve existing 248-block/4-MiB geometry.
- Large AR block/piece tuning does not materially improve the native ring.
- Small native AR LL/proxy settings trade tiny-message gains for regressions
  and have window variability; no global default change.
- Graph RS loses at some intermediate sizes; no universal graph speedup claim.
- Hybrid AR loses badly at tiny sizes and remains explicit/experimental.

Validation and publication:

- RS full curve: 2,940 timed samples, 21 sizes, every invocation checked.
- Final AG curve: 2,070 timed samples, 21 sizes, every invocation checked.
- Graph AG/RS: 1,680 timed samples, 11 sizes each, plus uneven tails.
- Hybrid AR: 420 timed samples, seven large sizes, plus tails and recheck.
- Native AR paired final: 60 timed samples; small tuning: 2,400 timed samples.
- Independent RS and hybrid signed-random checks cover uneven tails, rotating
  delayed ranks and groups of four asynchronous changing generations.
- Router policy 31 checks; dispatcher 28 checks; eager GPU reduction 26,569,136
  scalar/guard checks; router graph 1,000 captured mixed collectives and wrap.
- Makefile builds, clang-format 14 on new/changed code, diff whitespace and
  README links checked. Generated artifacts remain under build/.
- README curves/tables and standalone reproduction commands updated; full
  negative results, old curves, raw samples and final identities retained.

Boundaries and follow-on research:

No weaker publication fence, precision conversion, implicit runtime routing,
new CPU FP32 reduction backend, or hardware multicast claim was introduced.
Per-source packet readiness already exists; finer reader credits, persistent
application progress and new CPU reduction layouts remain research options
requiring profiling or a separately validated numerical/integration contract.
The current successful paths approach their measured transport references,
so those riskier rewrites are not prerequisites for publishing these gains.

No remote git push was performed. The user's request to put performance in
README was implemented in the shared working tree. Existing unrelated
untracked two-GPU experiments were left untouched.
