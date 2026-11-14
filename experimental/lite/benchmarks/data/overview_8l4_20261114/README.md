# Eight-L4 GPU and SHM overview figures

Each README overview contains **1-KiB–1-MiB latency** and **1-MiB–1-GiB bus
throughput** panels, with UCCL Lite and NCCL curves. BF16; AR in place,
AG/RS out of place. Sizes are complete input bytes/rank for AR/RS and complete
output bytes/rank for AG. Driver 580.178.04, CUDA 12.9 and NCCL 2.28.9.

## Current GPU measurements

The [four-to-eight-L4 transfer study](../transfer_4to8_20261114/README.md)
replaces GPU AR/RS through **4 MiB** and AG through **16 MiB**. Each point uses
its own experiment's paired NCCL control from the two reversed confirmation
sweeps: ten warmups and five rotated repetitions of twenty checked calls per
backend. The timer is the slowest-rank per-call CUDA-event interval, including
host submission gaps, after all producer streams complete. Inputs change every
call and every output/guard is checked. CPU 47 submits the dense calls.

The publication schedule is explicit and does not compute per-size minima:

| Collective | Updated range | Experiment / UCCL variant |
| --- | --- | --- |
| AR | 1 KiB–4 MiB | `tuning_ar / router-tuned` |
| RS | 1 KiB–4 MiB | `tuning_rs / router-tuned` |
| AG | 1–128 KiB | `tuning_ag / router-tuned` |
| AG | 256 KiB–1 MiB | `ag_attribution / mini` (16 blocks) |
| AG | 2–16 MiB | `ag_attribution / router-full-ag` (64-block cap) |

AR/RS retain the same configuration over their whole updated range, including
the AR 4-MiB regression against the base router. AG's region choices follow the
transfer/attribution study: mini-host adds a benefit at 256 KiB–1 MiB, while the
existing full GPU-progress router matches or beats it at larger sizes. No
NCCL value selects a UCCL path. These curves are a measured size-range schedule,
not a portable automatic-routing profile. Parallel submission regresses with
5-ms idle gaps; the underlying report retains those results.

GPU points above those ranges retain the recorded
[historical GPU overview](../collectives_8l4_20261111.csv): AR/RS from 8 MiB and
AG from 32 MiB through 1 GiB. They were not remeasured with the new settings.

## SHM and resident measurements

The SHM figures are unchanged: through **4 MiB**, they use the
[paired SHM refresh](../shm_refresh_8l4_20261114/README.md), with P2P disabled
and all NCCL channel edges verified as SHM. From **8 MiB through 1 GiB** they
retain the [SHM archive](../shm_8l4_20261111/README.md). Historical SHM AG reuses
the tuned UCCL host-packet curve and its separately measured NCCL SHM control.

The new `collectives-8l4-resident-small.svg` is separate from every overview.
It plots the fixed resident protocol versus its paired NCCL graph control at
all seven powers of two from 1 to 64 KiB, including AR's 64-KiB loss. Both use
input-ready **host-to-global-completion** timing. It neither substitutes those
numbers into the ordinary CUDA-event curves nor extrapolates resident
throughput beyond the measured range. See the transfer study for its prepared,
host-blocking execution contract and validation limits.

## Reproduction and provenance

[summary.csv](summary.csv) contains all **126 overview points**, with sources,
experiment/variant and historical/updated status per row. `build.py` reproduces
it; `plot.py --preview` writes the six SVGs into `docs/assets/` and PNG/PDF
previews into `build/figures/`. The SHM rows and SVGs remain byte-identical.
Run `python3 scripts/plot_compact_resident.py --preview` from the Lite root for
the resident figure. It reads the transfer study's confirmed host-timed rows.

Throughput is decimal GB/s: `bytes / latency_us / 1000`, scaled by 14/8 for AR
and 7/8 for AG/RS. Both overview panels include the 1-MiB point.
[provenance.json](provenance.json) records input hashes, update boundaries and
figure filenames; [checksums.json](checksums.json) covers this dataset. Earlier
GPU overview values remain in Git history and their original measurement
records; publishing the update does not rerun those benchmarks.
