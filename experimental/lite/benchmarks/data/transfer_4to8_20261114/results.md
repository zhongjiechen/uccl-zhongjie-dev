# Confirmed transfer results

All times are microseconds. Each confirmation point is the median of two sweep medians; each sweep has five repetitions of twenty checked calls, with ten warmups per backend/repetition. Variants rotate and the second sweep reverses their order. No best-per-size curve is selected. See [method and limitations](README.md).

## Specialized AllGather: ordinary event timing

Fixed 16-block mini-host geometry; out of place; complete output bytes/GPU. Positive reduction means faster than the current tuned router.

| Size | Tuned router | Mini-host | NCCL | Reduction vs tuned router |
| --- | ---: | ---: | ---: | ---: |
| 128 KiB | 70.38 | 70.66 | 129.31 | -0.4% |
| 256 KiB | 100.35 | 89.60 | 178.94 | 10.7% |
| 512 KiB | 151.55 | 133.11 | 251.65 | 12.2% |
| 1 MiB | 237.82 | 221.44 | 355.06 | 6.9% |
| 2 MiB | 450.56 | 405.47 | 560.38 | 10.0% |
| 4 MiB | 785.15 | 742.66 | 975.35 | 5.4% |
| 8 MiB | 1482.50 | 1428.48 | 1801.22 | 3.6% |
| 16 MiB | 2819.58 | 2815.49 | 3481.86 | 0.1% |

## AllGather attribution: existing GPU progress at larger sizes

A separate reversed paired experiment keeps GPU per-source progress enabled at every size, with either the regular 64-block cap or a 16-block cap. Mini-host remains fixed at 16 blocks. This isolates the new kernel from the existing progress option.

| Size | Limited-progress tuned router | Full GPU progress | Full GPU progress, 16-block cap | Mini-host | NCCL |
| --- | ---: | ---: | ---: | ---: | ---: |
| 128 KiB | 70.66 | 71.15 | 71.15 | 70.66 | 130.56 |
| 256 KiB | 101.12 | 100.10 | 100.86 | 90.11 | 180.22 |
| 512 KiB | 153.07 | 151.81 | 151.30 | 131.58 | 245.76 |
| 1 MiB | 236.54 | 237.57 | 246.02 | 222.21 | 342.78 |
| 2 MiB | 448.51 | 402.43 | 427.78 | 402.67 | 538.37 |
| 4 MiB | 782.59 | 742.91 | 814.84 | 753.92 | 941.06 |
| 8 MiB | 1475.32 | 1409.02 | 1522.43 | 1425.66 | 1748.73 |
| 16 MiB | 2806.26 | 2679.04 | 2886.14 | 2825.96 | 3391.23 |

## Prepared resident mode: matched host completion timing

AR is in place; AG/RS are out of place. All inputs are ready before the timer. Persistent workers are stopped in every control window. Graph NCCL is a single eight-device fork/join graph. These values are separate from event-timed ordinary calls.

| Operation | Size | Tuned router | Resident | NCCL graph | Eager NCCL | Tuned router / resident |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| AG | 1 KiB | 58.17 | 21.69 | 72.26 | 104.12 | 2.68× |
| AG | 8 KiB | 60.40 | 22.99 | 74.24 | 108.45 | 2.63× |
| AG | 16 KiB | 64.32 | 23.56 | 76.98 | 110.66 | 2.73× |
| AG | 32 KiB | 69.48 | 24.91 | 82.26 | 115.98 | 2.79× |
| AG | 64 KiB | 76.89 | 36.38 | 90.90 | 126.24 | 2.11× |
| RS | 1 KiB | 67.80 | 14.90 | 71.65 | 105.27 | 4.55× |
| RS | 8 KiB | 69.33 | 19.10 | 74.75 | 109.66 | 3.63× |
| RS | 16 KiB | 70.90 | 22.56 | 76.80 | 111.17 | 3.14× |
| RS | 32 KiB | 74.75 | 29.69 | 82.70 | 117.80 | 2.52× |
| RS | 64 KiB | 89.75 | 46.64 | 90.55 | 128.82 | 1.92× |
| AR | 1 KiB | 67.60 | 17.46 | 83.19 | 114.97 | 3.87× |
| AR | 8 KiB | 79.98 | 36.77 | 91.38 | 122.38 | 2.17× |
| AR | 16 KiB | 88.22 | 60.17 | 97.88 | 127.92 | 1.47× |
| AR | 32 KiB | 108.88 | 105.71 | 111.83 | 139.44 | 1.03× |
| AR | 64 KiB | 144.22 | 196.39 | 123.78 | 156.84 | 0.73× |

## Existing options: ordinary event timing

Base and every tuned variant use the same current source and common packet rules. The variants are defined in README.md. The one-CTA option applies only through 64 KiB; it is not a new path above that cutoff.

| Operation | Size | Base | Progress/compact | + Parallel launch | + One CTA | NCCL | Reduction: base to parallel |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| AG | 1 KiB | 64.50 | 54.27 | 41.72 | 40.19 | 87.30 | 35.3% |
| AG | 64 KiB | 83.46 | 73.21 | 60.42 | 66.80 | 110.59 | 27.6% |
| AG | 1 MiB | 272.13 | 241.15 | 237.06 | 237.31 | 356.84 | 12.9% |
| AG | 4 MiB | 784.38 | 783.87 | 782.59 | 783.85 | 973.31 | 0.2% |
| RS | 1 KiB | 69.11 | 65.54 | 51.19 | 49.92 | 92.93 | 25.9% |
| RS | 64 KiB | 109.82 | 84.99 | 72.96 | 69.89 | 115.44 | 33.6% |
| RS | 1 MiB | 343.81 | 323.33 | 317.44 | 319.74 | 338.94 | 7.7% |
| RS | 4 MiB | 983.30 | 982.78 | 969.98 | 969.73 | 1030.14 | 1.4% |
| AR | 1 KiB | 67.84 | 64.51 | 50.69 | 49.91 | 105.70 | 25.3% |
| AR | 64 KiB | 140.80 | 140.03 | 128.77 | 128.75 | 148.74 | 8.5% |
| AR | 1 MiB | 508.16 | 485.89 | 474.59 | 476.42 | 611.84 | 6.6% |
| AR | 4 MiB | 1520.38 | 1534.46 | 1559.30 | 1557.25 | 1879.30 | -2.6% |

## Sparse-call diagnostic: 5 ms host idle

One diagnostic sweep, three repetitions of ten calls; not pooled with confirmation. The idle interval precedes input generation and is excluded from host completion timing.

| Operation | Size | Progress | + Parallel launch | Resident | NCCL graph | Eager NCCL |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| AG | 1 KiB | 69.07 | 114.27 | 21.80 | 71.03 | 111.91 |
| AG | 64 KiB | 90.09 | 132.10 | 37.32 | 93.06 | 137.04 |
| RS | 1 KiB | 83.77 | 124.12 | 15.20 | 77.46 | 121.60 |
| RS | 64 KiB | 103.08 | 141.61 | 47.28 | 96.96 | 143.60 |
| AR | 1 KiB | 82.72 | 125.04 | 17.65 | 90.35 | 133.77 |
| AR | 64 KiB | 187.73 | 228.24 | 198.26 | 133.24 | 173.44 |
