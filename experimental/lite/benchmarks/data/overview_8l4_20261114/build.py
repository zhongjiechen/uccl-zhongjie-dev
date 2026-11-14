"""Combine confirmed small/medium results with recorded large-message data."""
import csv
from pathlib import Path

DATA = Path(__file__).resolve().parent
LITE = DATA.parents[2]
HISTORICAL_GPU = "benchmarks/data/collectives_8l4_20261111.csv"
HISTORICAL_SHM = "benchmarks/data/shm_8l4_20261111/figure_summary.csv"
AR_AG = "benchmarks/data/ar_ag_transfer_8l4_20261113/summary.csv"
RS = "benchmarks/data/rs_submit_8l4_20261113/final_summary.csv"
SHM = "benchmarks/data/shm_refresh_8l4_20261114/summary.csv"
OPERATIONS = ("all_reduce", "all_gather", "reduce_scatter")


def read(path):
    with (LITE / path).open() as stream:
        return list(csv.DictReader(stream))


def main():
    histories = {mode: {(row["collective"], int(row["bytes"])): row
                        for row in read(path)}
                 for mode, path in (("gpu", HISTORICAL_GPU), ("shm", HISTORICAL_SHM))}
    current = {}
    for operation, name, variant in (("all_reduce", "allreduce", "compact_combined"),
                                     ("all_gather", "allgather", "combined")):
        for row in read(AR_AG):
            if row["collective"] == name and row["variant"] == variant:
                current["gpu", operation, int(row["bytes"])] = (
                    float(row["median_us"]), float(row["paired_nccl_median_us"]),
                    AR_AG, variant)
    for row in read(RS):
        if row["variant"] == "hybrid_parallel":
            current["gpu", "reduce_scatter", int(row["bytes"])] = (
                float(row["median_us"]), float(row["paired_nccl_median_us"]),
                RS, "hybrid_parallel")
    for row in read(SHM):
        current["shm", row["collective"], int(row["bytes"])] = (
            float(row["uccl_median_us"]), float(row["nccl_median_us"]),
            SHM, "fixed-optimized-shm")
    assert len(current) == 78
    result = []
    for mode in ("gpu", "shm"):
        assert len(histories[mode]) == 63
        for operation in OPERATIONS:
            for size in (1 << power for power in range(10, 31)):
                old = histories[mode][operation, size]
                if size <= 4 << 20:
                    uccl, nccl, source, variant = current[mode, operation, size]
                    uccl_source = nccl_source = source
                    status = "updated"
                else:
                    uccl, nccl = float(old["uccl_median_us"]), float(old["nccl_median_us"])
                    source = HISTORICAL_GPU if mode == "gpu" else HISTORICAL_SHM
                    uccl_source = nccl_source = source
                    if mode == "shm":
                        uccl_source = f"benchmarks/data/{old['uccl_source']}/summary.csv"
                        nccl_source = f"benchmarks/data/{old['nccl_source']}/summary.csv"
                    variant = old["uccl_path"]
                    status = "historical"
                result.append(dict(testbed=mode, collective=operation, bytes=size,
                                   placement=old["placement"], uccl_median_us=uccl,
                                   nccl_median_us=nccl, uccl_source=uccl_source,
                                   nccl_source=nccl_source, variant=variant, status=status))
    with (DATA / "summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=result[0].keys(), lineterminator="\n")
        writer.writeheader()
        writer.writerows(result)


if __name__ == "__main__":
    main()
