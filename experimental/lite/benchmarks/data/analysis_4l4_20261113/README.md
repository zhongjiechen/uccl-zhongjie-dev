# Four-L4 analysis checks, November 13, 2026

See [the analysis](../../../docs/4l4-performance-analysis.md) for scope, bounds,
hardware, dtype, selection and limitations. These are short local confirmation
runs, not replacements for the longer historical paired sweeps.

Commands executed sequentially on idle `mibura-sky-test-02`:

```bash
make hardware_profiler peer_ag_rs allreduce
./build/uccl_lite_hardware_profiler --devices 0,1,2,3 \
  --peer-order 0,1,3,2 --gpu-bytes 16777216 --cpu-bytes 8388608 \
  --iterations 10 --warmups 2 --repeats 3 --threads-per-node 4 > hardware.txt
./build/uccl_lite_peer_ag_rs -b 67108864 -e 1073741824 \
  -w 2 -n 5 -r 3 > peer_large.log
./build/uccl_lite_allreduce -b 1073741824 -e 1073741824 \
  -w 2 -n 3 > ar_1g.log
make peer-ag-rs-packet
./build/uccl_lite_peer_ag_rs_packet --packet-router-ag \
  --packet-router-per-source -b 1073741824 -e 1073741824 \
  -w 2 -n 5 -r 3 > packet_1g.log
```

Paths above are from the lite directory; outputs were redirected into this
record directory. `identity.json` records revision/source/library and binary
hashes; the packet binary is updated after its Makefile rebuild.

AR is FP32 in place, one forked process per GPU, maximum-rank host call latency.
AG/RS are FP32 out of place, one process, maximum-GPU event latency including
host submission gaps. Both harnesses check full output outside timing on every
warmup/timed collective. AR's null control is intentionally not checked.
Peer baseline is linked NCCL 2.28.7 default. There is no paired NCCL AR run here.
AR reports means of 3 calls; peer reports median of 3 run means of 5 calls.
Raw variants, including slower paths, remain in logs. Packet and peer selection
are harness candidate comparisons; never pool their NCCL samples across runs.
