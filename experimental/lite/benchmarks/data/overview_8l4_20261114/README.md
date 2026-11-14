# Eight-L4 GPU and SHM overview figures

Each README figure contains **1-KiB–1-MiB latency** and **1-MiB–1-GiB bus
throughput** panels, with only UCCL Lite and NCCL curves. BF16; AR in place,
AG/RS out of place. Sizes are complete input bytes/rank for AR/RS and complete
output bytes/rank for AG. Driver 580.178.04, CUDA 12.9 and NCCL 2.28.9.

Sources through **4 MiB** are the fixed-configuration, two-sweep confirmations:

- GPU AR/AG: [scheduling transfer](../ar_ag_transfer_8l4_20261113/README.md),
  `compact_combined` AR and `combined` AG.
- GPU RS: [submission follow-up](../rs_submit_8l4_20261113/provenance.json),
  `hybrid_parallel`, measured on its recorded earlier source revision.
- SHM AR/AG/RS: [fresh paired SHM sweeps](../shm_refresh_8l4_20261114/README.md),
  with P2P disabled and all NCCL channel edges verified as SHM.

Sources from **8 MiB through 1 GiB** are the recorded
[GPU overview](../collectives_8l4_20261111.csv) and
[SHM archive](../shm_8l4_20261111/README.md). Large-message points are historical
and use their recorded settings; they were not remeasured with the new small
message options. Historical SHM AG reuses the tuned UCCL host-packet curve and
has a separately measured NCCL SHM control, as documented in that archive.

All timings use slowest-rank CUDA events including host submission gaps.
The updated points use ten warmups and five alternating repetitions of twenty
checked calls per backend, summarized across two sweeps; historical protocols
are in their linked records. The update boundary is fixed at 4 MiB and does not
select the fastest variant per size. These composite curves are a presentation
of recorded configurations, not an automatic-routing calibration profile.

[summary.csv](summary.csv) contains all **126 plotted size points** and identifies
both sources and the historical/updated status per row. `build.py` reproduces
it from those sources; `plot.py --preview` writes the six SVGs into `docs/assets/`
and PNG/PDF previews into `build/figures/`. Throughput is decimal GB/s, using
`bytes / latency / 1000`, scaled by 14/8 for AR and 7/8 for AG/RS.
[provenance.json](provenance.json) records input hashes and figure filenames;
[checksums.json](checksums.json) covers this dataset.
