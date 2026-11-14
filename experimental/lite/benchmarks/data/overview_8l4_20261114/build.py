"""Build all six overviews exclusively from fresh full-range GPU/SHM sweeps."""
import csv
import math
from pathlib import Path

DATA = Path(__file__).resolve().parent
LITE = DATA.parents[2]
SOURCE = "benchmarks/data/full_refresh_8l4_20261114/summary.csv"
OPERATIONS = ("all_reduce", "all_gather", "reduce_scatter")


def main():
    with (LITE / SOURCE).open() as stream:
        rows = list(csv.DictReader(stream))
    expected = {(mode, op, 1 << power) for mode in ("gpu", "shm")
                for op in OPERATIONS for power in range(10, 31)}
    actual = {(r["testbed"], r["collective"], int(r["bytes"])) for r in rows}
    if actual != expected or len(rows) != len(expected):
        raise ValueError("full refresh requires all 126 unique workload keys")
    result = []
    for row in rows:
        if row["status"] != "fresh-full-sweep":
            raise ValueError("historical or unconfirmed point in full refresh")
        for field in ("uccl_median_us", "nccl_median_us"):
            value = float(row[field])
            if not math.isfinite(value) or value <= 0:
                raise ValueError("invalid measured latency")
        result.append(dict(testbed=row["testbed"], collective=row["collective"],
                           bytes=row["bytes"], placement=row["placement"],
                           uccl_median_us=row["uccl_median_us"],
                           nccl_median_us=row["nccl_median_us"],
                           uccl_source=SOURCE, nccl_source=SOURCE,
                           variant=row["variant"], status=row["status"]))
    with (DATA / "summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=result[0].keys(), lineterminator="\n")
        writer.writeheader()
        writer.writerows(result)


if __name__ == "__main__":
    main()
