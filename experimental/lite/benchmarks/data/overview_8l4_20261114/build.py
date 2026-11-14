"""Combine confirmed small/medium results with recorded large-message data."""
import csv
import math
from pathlib import Path

DATA = Path(__file__).resolve().parent
LITE = DATA.parents[2]
HISTORICAL_GPU = "benchmarks/data/collectives_8l4_20261111.csv"
HISTORICAL_SHM = "benchmarks/data/shm_8l4_20261111/figure_summary.csv"
TRANSFER = "benchmarks/data/transfer_4to8_20261114/summary.csv"
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
    transfer = {}
    for row in read(TRANSFER):
        key = row["experiment"], int(row["bytes"]), row["variant"]
        if key in transfer:
            raise ValueError(f"duplicate transfer measurement: {key}")
        transfer[key] = row
    for operation, suffix in (("all_reduce", "ar"), ("all_gather", "ag"),
                              ("reduce_scatter", "rs")):
        end_power = 24 if operation == "all_gather" else 22
        for power in range(10, end_power + 1):
            size = 1 << power
            experiment, variant = f"tuning_{suffix}", "router-tuned"
            # Fixed publication schedule from the checked attribution study;
            # do not choose each point's minimum or use the NCCL timings.
            if operation == "all_gather" and size >= 256 << 10:
                experiment = "ag_attribution"
                variant = "mini" if size <= 1 << 20 else "router-full-ag"
            pair = [transfer[experiment, size, name] for name in (variant, "nccl")]
            for row in pair:
                if row["timing"] != "slowest-GPU-event" or int(row["sweeps"]) != 2:
                    raise ValueError(f"unmatched timing/confirmation: {experiment}, {size}")
                value = float(row["median_us"])
                if not math.isfinite(value) or value <= 0:
                    raise ValueError("invalid transfer latency")
            current["gpu", operation, size] = (
                *(float(row["median_us"]) for row in pair),
                TRANSFER, f"{experiment}/{variant}")
    for row in read(SHM):
        current["shm", row["collective"], int(row["bytes"])] = (
            float(row["uccl_median_us"]), float(row["nccl_median_us"]),
            SHM, "fixed-optimized-shm")
    assert len(current) == 80
    result = []
    for mode in ("gpu", "shm"):
        assert len(histories[mode]) == 63
        for operation in OPERATIONS:
            for size in (1 << power for power in range(10, 31)):
                old = histories[mode][operation, size]
                if (mode, operation, size) in current:
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
