# Machine types and thresholds

Run `chs list machine-types` to view supported machine types in the terminal, and see [Tests and suites](../tests/README.md) for what each test measures.

## Supported machine types

### TPU

| Machine type | Accelerator | Kubernetes | Slurm | Standalone (TPU VM) |
| --- | --- | --- | --- | --- |
| `ct6e-standard-4t` | TPU v6e (Trillium) | Yes | No | Yes |
| `tpu7x-standard-4t` | TPU7x (Ironwood) | Yes | No | Yes |

### GPU

| Machine type | Accelerator | Kubernetes | Slurm |
| --- | --- | --- | --- |
| `a3-megagpu-8g` | NVIDIA H100 80GB | Yes | Yes |
| `a3-ultragpu-8g` | NVIDIA H200 141GB | Yes | Yes |
| `a4-highgpu-8g` | NVIDIA B200 | Yes | Yes |
| `a4x-highgpu-4g` | NVIDIA GB200 NVL72 | Yes | Yes |
| `a4x-maxgpu-4g-metal` | NVIDIA GB300 NVL72 | Yes | Yes |
| `a5x-highgpu-4g-metal` | NVIDIA Vera Rubin NVL72 | Yes | No |

All nodes targeted by a single `chs run` invocation must share the same
accelerator (machine type). CHS determines it automatically from the targeted
nodes.

## Thresholds

Pass/fail thresholds are defined per machine type in each test's
`thresholds.yaml` file:

### TPU tests

- [`tpu-info`](../tests/tpu-info/thresholds.yaml)
- [`tpu-gemm`](../tests/tpu-gemm/thresholds.yaml)
- [`tpu-hbm`](../tests/tpu-hbm/thresholds.yaml)
- [`tpu-host-device`](../tests/tpu-host-device/thresholds.yaml)
- [`tpu-collectives`](../tests/tpu-collectives/thresholds.yaml)
- [`tpu-compute-stress`](../tests/tpu-compute-stress/thresholds.yaml)
- [`tpu-hbm-stress`](../tests/tpu-hbm-stress/thresholds.yaml)
- [`tpu-network-stress`](../tests/tpu-network-stress/thresholds.yaml)

### GPU tests

- [`nvidia-smi`](../tests/nvidia-smi/thresholds.yaml)
- [`dcgm`](../tests/dcgm/thresholds.yaml)
- [`gemm`](../tests/gemm/thresholds.yaml)
- [`stream`](../tests/stream/thresholds.yaml)
- [`nccl-single-node`](../tests/nccl-single-node/thresholds.yaml)
- [`nccl-cluster`](../tests/nccl-cluster/thresholds.yaml)
- [`nccl-pairwise`](../tests/nccl-pairwise/thresholds.yaml)
- [`nccl-domain`](../tests/nccl-domain/thresholds.yaml)
