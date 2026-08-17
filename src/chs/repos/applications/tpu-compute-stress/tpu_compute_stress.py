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

import os
import time
import logging
import functools

init_args = (
    " --tpu_link_up_check_timeout=15s"
    " --megascale_error_reporter_abort_on_hang=true"
)
os.environ["LIBTPU_INIT_ARGS"] = (
    os.environ.get("LIBTPU_INIT_ARGS", "") + init_args
)

# IMPORTANT: These imports must sit below the LIBTPU_INIT_ARGS population block
# because JAX reads the environment configs directly upon initialization.
import numpy as np
import jax
import jax.numpy as jnp

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


MATRIX_DIM = 16384
NUM_ITERATIONS_WARMUP = 5
NUM_ITERATIONS_MEASURE = 1500
NUM_MATMULS_PER_ITERATION = 4


@functools.partial(jax.pmap, axis_name="device")
def init_payloads(key):
    k1, k2, k3 = jax.random.split(key, 3)
    A = jax.random.normal(k1, (MATRIX_DIM, MATRIX_DIM), dtype=jnp.bfloat16)

    W1 = jax.random.normal(
        k2, (MATRIX_DIM, MATRIX_DIM), dtype=jnp.bfloat16
    ) / jnp.sqrt(MATRIX_DIM)
    W2 = jax.random.normal(
        k3, (MATRIX_DIM, MATRIX_DIM), dtype=jnp.bfloat16
    ) / jnp.sqrt(MATRIX_DIM)

    return A, W1, W2


@functools.partial(jax.pmap, axis_name="device", in_axes=(0, 0, 0, None))
def hardware_health_stress(A, W1, W2, iterations):
    def body_fn(i, val):
        x = val

        for _ in range(NUM_MATMULS_PER_ITERATION):
            x = jnp.matmul(x, W1)
            x = jnp.matmul(x, W2)

        x = jax.lax.add(x, jnp.roll(x, shift=1, axis=0))

        x = jax.nn.swish(x)

        mean_sq = jnp.mean(jnp.square(x), axis=-1, keepdims=True)
        x = x * jax.lax.rsqrt(mean_sq + 1e-5)

        return x

    return jax.lax.fori_loop(0, iterations, body_fn, A)


def main():
    logging.info("Starting TPU Compute stress script...")
    logging.info(
        f"MATRIX_DIM={MATRIX_DIM} NUM_ITERATIONS_MEASURE={NUM_ITERATIONS_MEASURE}"
    )
    local_devices = jax.local_devices()
    node_name = os.environ.get("MY_NODE_NAME", "UnknownHost")

    logging.info(f"NODE:{node_name} EVENT:INIT_START")

    logging.info(
        f"Process {jax.process_index()} on {node_name} | Local Devices: {len(local_devices)}"
    )

    root_key = jax.random.PRNGKey(42)
    device_keys = jnp.stack([root_key] * len(local_devices))

    logging.info(f"NODE:{node_name} EVENT:WARMUP_START")
    A_stack, W1_stack, W2_stack = init_payloads(device_keys)

    warmup_iters = jnp.array(NUM_ITERATIONS_WARMUP, dtype=jnp.int32)
    measure_iters = jnp.array(NUM_ITERATIONS_MEASURE, dtype=jnp.int32)

    warmup_res = hardware_health_stress(
        A_stack, W1_stack, W2_stack, warmup_iters
    )
    warmup_res.block_until_ready()

    logging.info(f"NODE:{node_name} EVENT:MEASURE_START")
    start_time = time.time()
    outputs = hardware_health_stress(A_stack, W1_stack, W2_stack, measure_iters)
    outputs.block_until_ready()
    duration = time.time() - start_time

    flops_per_matmul = 2.0 * (MATRIX_DIM**3)
    total_flops_per_loop = flops_per_matmul * (NUM_MATMULS_PER_ITERATION * 2)
    total_flops = total_flops_per_loop * NUM_ITERATIONS_MEASURE
    tflops_per_core = (total_flops / duration) / (1e12)

    cpu_outputs = np.asarray(outputs)
    sdc_failed = False

    baseline = cpu_outputs[0]

    for i in range(len(local_devices)):
        if not np.array_equal(cpu_outputs[i], baseline):
            sdc_failed = True
            break

    if sdc_failed:
        logging.error(
            f"NODE:{node_name} EVENT:STRESS_FAIL. SDC (Bit flips) Detected"
        )
    else:
        logging.info(
            f"NODE:{node_name} EVENT:MEASURE_END DURATION:{duration:.2f}s TFLOPs: {tflops_per_core:.2f}"
        )
        logging.info(f"NODE:{node_name} EVENT:STRESS_PASS")


if __name__ == "__main__":
    main()
