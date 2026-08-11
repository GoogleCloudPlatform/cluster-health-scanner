# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import functools
import os
import time
import logging

accel_type = os.environ.get("ACCELERATOR_TYPE", "").lower()

# v6e (Trillium) and v5e use efficiency dies maxing at P_STATE=3. v4/v5p/v7x use max-power dies at P_STATE=7
EFFICIENCY_TPU_ARCHITECTURES = ("v6e", "v5e", "ct6e")
p_state = (
    "3"
    if any(arch in accel_type for arch in EFFICIENCY_TPU_ARCHITECTURES)
    else "7"
)

init_args = (
    " --xla_tpu_scoped_vmem_limit_kib=65536"
    " --xla_jf_bounds_check=false"
    f" --xla_tpu_dvfs_p_state={p_state}"
    " --tpu_link_up_check_timeout=15s"
    " --megascale_error_reporter_abort_on_hang=true"
)
os.environ["LIBTPU_INIT_ARGS"] = (
    os.environ.get("LIBTPU_INIT_ARGS", "") + init_args
)

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

try:
    _hbm_bytes = (
        jax.local_devices()[0].memory_stats().get("bytes_limit", 32 * 1024**3)
    )
except Exception:  # Fallback if memory_stats fails or backend is unavailable
    _hbm_bytes = 32 * 1024**3

# Target ~20% of available HBM to avoid OOM with multiple buffer copies
_safe_elements = int((_hbm_bytes * 0.20) // 4)
# Snapping _safe_elements strictly down to the nearest floor power-of-two
# to natively sidestep the XLA compilation padding explosion on threefry/randint
NUM_ELEMENTS = 1 << (_safe_elements.bit_length() - 1)
NUM_ITERATIONS_WARMUP = 1
NUM_ITERATIONS_MEASURE = 15000


@jax.pmap
def init_memory(key):
    return jax.random.randint(
        key, (NUM_ELEMENTS,), minval=0, maxval=2147483647, dtype=jnp.uint32
    )


@functools.partial(jax.pmap, in_axes=(0, None))
def hbm_burn(a, num_iters):
    def body_fun(i, x):
        return jax.lax.bitwise_not(x)

    return jax.lax.fori_loop(0, num_iters, body_fun, a)


@jax.pmap
def hbm_checksum(a):
    a_int64 = a.astype(jnp.int64)
    return jnp.sum(a_int64 ^ jnp.roll(a_int64, shift=1), dtype=jnp.int64)


def main():
    logging.info("Starting TPU HBM stress script...")
    logging.info(f"ACCELERATOR_TYPE: {accel_type}, Deduced P_STATE: {p_state}")
    logging.info(
        f"NUM_ELEMENTS={NUM_ELEMENTS} NUM_ITERATIONS_MEASURE={NUM_ITERATIONS_MEASURE}"
    )
    local_devices = jax.local_devices()
    node_name = os.environ.get("MY_NODE_NAME", "UnknownHost")

    logging.info(f"NODE:{node_name} EVENT:INIT_START")
    logging.info(
        f"Process {jax.process_index()} on {node_name} | Local Devices: {len(local_devices)}"
    )

    logging.info(f"NODE:{node_name} EVENT:WARMUP_START")
    root_key = jax.random.PRNGKey(42)
    device_keys = jnp.stack([root_key] * len(local_devices))
    A_mem = init_memory(device_keys)

    warmup_res = hbm_burn(A_mem, NUM_ITERATIONS_WARMUP)
    jax.block_until_ready(warmup_res)

    logging.info(f"NODE:{node_name} EVENT:MEASURE_START")
    start_time = time.time()

    outputs = hbm_burn(A_mem, NUM_ITERATIONS_MEASURE)
    jax.block_until_ready(outputs)

    duration = time.time() - start_time

    bytes_per_iteration = (NUM_ELEMENTS * 4) * 2
    total_bytes_transferred = bytes_per_iteration * NUM_ITERATIONS_MEASURE
    gb_per_second = (total_bytes_transferred / duration) / (1e9)

    devices_per_chip = max(1, len(local_devices) // 4)
    chip_bandwidth = gb_per_second * devices_per_chip

    logging.info(
        f"NODE:{node_name} EVENT:MEASURE_END DURATION:{duration:.2f}s BANDWIDTH_GBPS:{chip_bandwidth:.2f}"
    )

    checksums = jax.device_get(hbm_checksum(outputs))
    golden_checksum = checksums[0]
    is_valid = True

    for i, dev in enumerate(local_devices):
        if checksums[i] != golden_checksum:
            is_valid = False
            break

    if not is_valid:
        logging.error(
            f"NODE:{node_name} EVENT:STRESS_FAIL. TPU devices failed SDC check"
        )
    else:
        logging.info(f"NODE:{node_name} EVENT:STRESS_PASS")


if __name__ == "__main__":
    main()
