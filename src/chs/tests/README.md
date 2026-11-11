# Tests and suites

CHS organizes diagnostics into **suites** (predefined testing bundles, `--suite lite` or `--suite standard`) and **tests** (individual checks). When running a suite, CHS automatically runs only the tests supported on the detected accelerator (machine type) and scheduler. Any test can also be run individually or grouped outside of a suite using `--tests <name>[,<name>...]`.

## TPU tests

| Test name | Description | Suites |
| --- | --- | --- |
| [`tpu-info`](tpu-info) | Basic TPU health covering device count, `libtpu` version, and TPU device type. | `lite`, `standard` |
| [`tpu-gemm`](tpu-gemm) | Performs generalized matrix multiplications (GEMM) through the TPU to calculate exactly how many peak TFLOP/s the compute pipeline can sustain. | `lite`, `standard` |
| [`tpu-hbm`](tpu-hbm) | Generates performance traces to measure the absolute architectural read and write bandwidth limits of the HBM. | `lite`, `standard` |
| [`tpu-host-device`](tpu-host-device) | Measures host-to-device and device-to-host transfer bandwidth (GB/s) between the host CPU and the TPU. | `lite`, `standard` |
| [`tpu-collectives`](tpu-collectives) | Times distributed cross-chip operations (like All-Reduce and All-Gather) to measure the real-world bandwidth limits of the interconnect infrastructure. | `lite`, `standard` |
| [`tpu-compute-stress`](tpu-compute-stress) | Maximizes the TPU's core math engines with intense, sustained matrix multiplications to verify system-level computational stability and expose any silent data corruption. | `standard` |
| [`tpu-hbm-stress`](tpu-hbm-stress) | Checks the bottlenecks of the High Bandwidth Memory (HBM) using rapid integer operations to strictly measure maximum memory bandwidth and validate data integrity. | `standard` |
| [`tpu-network-stress`](tpu-network-stress) | Continuously swaps massive random payloads across the TPU high-speed interconnect (ICI) links to measure peak sustained network bandwidth and validate health against bit-flips. | `standard` |

## GPU tests

| Test name | Description | Suites |
| --- | --- | --- |
| [`nvidia-smi`](nvidia-smi) | Queries GPU status to measure power draw, clock frequencies, and performance states, while verifying driver health and GPU count. | `lite`, `standard` |
| [`dcgm`](dcgm) | Runs DCGM diagnostic suites to stress-test PCIe, memory, and compute engines, measuring hardware stability and identifying faults, thermal issues, or power throttling. | `lite`, `standard` |
| [`gemm`](gemm) | Runs General Matrix Multiply (GEMM) benchmarks to measure peak floating-point compute throughput (TFLOPS) and verify computation correctness. | `lite`, `standard` |
| [`stream`](stream) | Measures sustainable GPU memory bandwidth (GB/s) using the STREAM benchmark to verify memory subsystem throughput. | `standard` |
| [`nccl-single-node`](nccl-single-node) | Runs NCCL benchmarks within a single node to measure intra-node GPU-to-GPU bandwidth (GB/s) and latency. | `lite` |
| [`nccl-cluster`](nccl-cluster) | Runs NCCL benchmarks across multiple nodes to measure inter-node bandwidth (GB/s) and latency under collective workloads. | `standard` |
| [`nccl-pairwise`](nccl-pairwise) | Runs NCCL benchmarks between node pairs to measure point-to-point bandwidth (GB/s) and latency, helping isolate degraded network links. | `standard` |
| [`nccl-domain`](nccl-domain) | Runs NCCL benchmarks separately within each NVLink domain (IMEX/NVLD racks) to measure intra-domain MNNVL bandwidth (GB/s), isolating a degraded domain without invalidating the rest of the cluster. | None (use `--tests`) |
