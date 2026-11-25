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

import numpy as np
import jax
import jax.numpy as jnp

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


MATRIX_DIM = 16384
MATMUL_STEPS_PER_DISPATCH = 5
NUM_ITERATIONS_WARMUP = MATMUL_STEPS_PER_DISPATCH
NUM_ITERATIONS_MEASURE = 1500
NUM_MATMULS_PER_ITERATION = 4
RMS_NORM_EPS = 1e-5
RANDOM_SEED = 42

LOW_PRECISION_SECONDS = 45
LOW_PRECISION_STEPS_PER_DISPATCH = 120

VECTOR_SECONDS = 20
VECTOR_STEPS_PER_DISPATCH = 50_000
VECTOR_SHAPE = (8192, 128)


def _local_mesh():
    return jax.sharding.Mesh(np.array(jax.local_devices()), axis_names="device")


def _replicated_from_host(host_array):
    replicated = jax.sharding.NamedSharding(
        _local_mesh(), jax.sharding.PartitionSpec()
    )
    return jax.device_put(host_array, replicated)


def _on_every_device(fn, *arrays):
    replicated = jax.sharding.PartitionSpec()
    return jax.shard_map(
        fn,
        mesh=_local_mesh(),
        in_specs=(replicated,) * len(arrays),
        out_specs=replicated,
    )(*arrays)


def _per_device_results(x):
    return [shard.data for shard in x.addressable_shards]


def _run_until_deadline(seconds, dispatch, x):
    dispatches = 0
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        x = jax.block_until_ready(dispatch(x))
        dispatches += 1
    return x, dispatches


@jax.jit
def _init_single_device_payloads(key):
    k1, k2, k3 = jax.random.split(key, 3)
    A = jax.random.normal(k1, (MATRIX_DIM, MATRIX_DIM), dtype=jnp.bfloat16)

    W1 = jax.random.normal(
        k2, (MATRIX_DIM, MATRIX_DIM), dtype=jnp.bfloat16
    ) / jnp.sqrt(MATRIX_DIM)
    W2 = jax.random.normal(
        k3, (MATRIX_DIM, MATRIX_DIM), dtype=jnp.bfloat16
    ) / jnp.sqrt(MATRIX_DIM)

    return A, W1, W2


def init_payloads(key):
    local_devices = jax.local_devices()
    sharding = jax.sharding.NamedSharding(
        _local_mesh(), jax.sharding.PartitionSpec()
    )
    with jax.default_device(local_devices[0]):
        a0, w1_0, w2_0 = _init_single_device_payloads(key)
        a_np, w1_np, w2_np = jax.device_get((a0, w1_0, w2_0))

    def _replicate(d0_arr, host_np):
        shards = [d0_arr] + [
            jax.device_put(host_np, d) for d in local_devices[1:]
        ]
        return jax.make_array_from_single_device_arrays(
            (MATRIX_DIM, MATRIX_DIM), sharding, shards
        )

    A = _replicate(a0, a_np)
    W1 = _replicate(w1_0, w1_np)
    W2 = _replicate(w2_0, w2_np)
    jax.block_until_ready((A, W1, W2))
    return A, W1, W2


def _rms_normalise(x):
    mean_sq = jnp.mean(jnp.square(x), axis=-1, keepdims=True)
    return x * jax.lax.rsqrt(mean_sq + RMS_NORM_EPS)


def _stress_step(x, W1, W2):
    for _ in range(NUM_MATMULS_PER_ITERATION):
        x = jnp.matmul(x, W1)
        x = jnp.matmul(x, W2)

    x = jax.lax.add(x, jnp.roll(x, shift=1, axis=0))

    x = jax.nn.swish(x)

    return _rms_normalise(x)


@functools.partial(jax.jit, static_argnums=(3,))
def _stress_device_loop(A, W1, W2, iterations):
    def _loop(x, w1, w2):
        return jax.lax.fori_loop(
            0, iterations, lambda _, v: _stress_step(v, w1, w2), x, unroll=True
        )

    return _on_every_device(_loop, A, W1, W2)


def hardware_health_stress(A, W1, W2, iterations):
    remaining = int(iterations)
    x = A
    while remaining > 0:
        step = min(MATMUL_STEPS_PER_DISPATCH, remaining)
        x = _stress_device_loop(x, W1, W2, step)
        remaining -= step
    return _per_device_results(x)


def low_precision_dtype(device_kind):
    return jnp.float8_e4m3fn if "7x" in device_kind.lower() else jnp.int8


def _is_integer(dtype):
    return jnp.issubdtype(dtype, jnp.integer)


def _largest_value(dtype):
    if _is_integer(dtype):
        return float(jnp.iinfo(dtype).max)
    return float(jnp.finfo(dtype).max)


def _fit_into(values, dtype):
    peak = jnp.maximum(jnp.max(jnp.abs(values)), jnp.finfo(jnp.float32).tiny)
    scaled = values / peak * _largest_value(dtype)
    if _is_integer(dtype):
        scaled = jnp.round(scaled)
    return scaled.astype(dtype)


def _low_precision_step(x, w):
    accumulator = jnp.int32 if _is_integer(x.dtype) else jnp.float32
    product = jnp.matmul(x, w, preferred_element_type=accumulator)
    return _fit_into(product.astype(jnp.float32), x.dtype)


@functools.partial(jax.jit, static_argnums=(2,))
def _fit_inputs_into(A, W, dtype):
    def _fit(a, w):
        return (
            _fit_into(a.astype(jnp.float32), dtype),
            _fit_into(w.astype(jnp.float32), dtype),
        )

    return _on_every_device(_fit, A, W)


@functools.partial(jax.jit, static_argnums=(2,))
def _low_precision_dispatch(x, w, steps):
    def _loop(x, w):
        return jax.lax.fori_loop(
            0, steps, lambda _, v: _low_precision_step(v, w), x
        )

    return _on_every_device(_loop, x, w)


def hardware_low_precision_stress(A, W, dtype, seconds):
    x, w = _fit_inputs_into(A, W, dtype)
    x, dispatches = _run_until_deadline(
        seconds,
        lambda x: _low_precision_dispatch(
            x, w, LOW_PRECISION_STEPS_PER_DISPATCH
        ),
        x,
    )
    return _per_device_results(x), dispatches


def _wrap_into_half_to_three_halves(values):
    return 0.5 + jnp.mod(values, 1.0)


def _vector_step(step, x):
    key = jax.random.fold_in(jax.random.PRNGKey(RANDOM_SEED), step)
    noise = jax.random.uniform(key, x.shape)
    curve = jnp.exp(x) + jnp.tanh(x) + jax.lax.rsqrt(x) + jnp.log(x)
    return _wrap_into_half_to_three_halves(curve + noise)


@functools.partial(jax.jit, static_argnums=(1,))
def _vector_dispatch(x, steps):
    return _on_every_device(
        lambda x: jax.lax.fori_loop(0, steps, _vector_step, x), x
    )


def hardware_vector_stress(seconds):
    x, dispatches = _run_until_deadline(
        seconds,
        lambda x: _vector_dispatch(x, VECTOR_STEPS_PER_DISPATCH),
        _replicated_from_host(np.ones(VECTOR_SHAPE, np.float32)),
    )
    return _per_device_results(x), dispatches


def devices_differing_from_device_0(per_device_results):
    host_bytes = [
        np.asarray(result).view(np.uint8)
        for result in jax.device_get(per_device_results)
    ]
    return [
        index
        for index, device_bytes in enumerate(host_bytes)
        if not np.array_equal(device_bytes, host_bytes[0])
    ]


def devices_with_non_finite_values(per_device_results):
    return [
        index
        for index, result in enumerate(jax.device_get(per_device_results))
        if not np.isfinite(np.asarray(result, dtype=np.float32)).all()
    ]


def phase_results_valid(node_name, phase, per_device_results):
    differing = devices_differing_from_device_0(per_device_results)
    non_finite = devices_with_non_finite_values(per_device_results)
    if differing:
        logging.error(
            f"NODE:{node_name} EVENT:RESULTS_DIFFER_FROM_DEVICE_0 "
            f"PHASE:{phase} DEVICES:{','.join(map(str, differing))}"
        )
    if non_finite:
        logging.error(
            f"NODE:{node_name} EVENT:NON_FINITE_RESULTS "
            f"PHASE:{phase} DEVICES:{','.join(map(str, non_finite))}"
        )
    return not differing and not non_finite


def matmul_results_match(per_device_results):
    cpu_outputs = np.asarray(jax.device_get(per_device_results))
    baseline = cpu_outputs[0]
    return all(np.array_equal(output, baseline) for output in cpu_outputs)


def experimental_enabled():
    return os.environ.get("CHS_EXPERIMENTAL") == "true"


def report_results(node_name, results_valid, duration, tflops_per_core):
    if results_valid:
        logging.info(
            f"NODE:{node_name} EVENT:MEASURE_END DURATION:{duration:.2f}s TFLOPs: {tflops_per_core:.2f}"
        )
        logging.info(f"NODE:{node_name} EVENT:STRESS_PASS")
    else:
        logging.error(
            f"NODE:{node_name} EVENT:STRESS_FAIL. SDC (Bit flips) Detected"
        )


def stress_other_units_and_check_results(
    node_name, device_kind, A_stack, W1_stack, matmul_outputs
):
    dtype = low_precision_dtype(device_kind)
    logging.info(
        f"NODE:{node_name} EVENT:LOW_PRECISION_START DTYPE:{jnp.dtype(dtype).name}"
    )
    low_precision_start_time = time.time()
    low_precision_outputs, low_precision_dispatches = (
        hardware_low_precision_stress(
            A_stack, W1_stack, dtype, LOW_PRECISION_SECONDS
        )
    )
    logging.info(
        f"NODE:{node_name} EVENT:LOW_PRECISION_END "
        f"DISPATCHES:{low_precision_dispatches} "
        f"STEPS_PER_DISPATCH:{LOW_PRECISION_STEPS_PER_DISPATCH} "
        f"DURATION_WITH_COMPILE:{time.time() - low_precision_start_time:.2f}s"
    )

    logging.info(f"NODE:{node_name} EVENT:VECTOR_START")
    vector_start_time = time.time()
    vector_outputs, vector_dispatches = hardware_vector_stress(VECTOR_SECONDS)
    logging.info(
        f"NODE:{node_name} EVENT:VECTOR_END DISPATCHES:{vector_dispatches} "
        f"STEPS_PER_DISPATCH:{VECTOR_STEPS_PER_DISPATCH} "
        f"DURATION_WITH_COMPILE:{time.time() - vector_start_time:.2f}s"
    )

    phases = {
        "matmul": matmul_outputs,
        "low_precision": low_precision_outputs,
        "vector": vector_outputs,
    }
    phase_results = [
        phase_results_valid(node_name, phase, outputs)
        for phase, outputs in phases.items()
    ]
    return all(phase_results)


def main():
    node_name = os.environ.get("MY_NODE_NAME", "UnknownHost")

    logging.info("Starting TPU Compute stress script...")
    logging.info(
        f"MATRIX_DIM={MATRIX_DIM} NUM_ITERATIONS_MEASURE={NUM_ITERATIONS_MEASURE}"
    )
    logging.info(f"LIBTPU_INIT_ARGS={os.environ.get('LIBTPU_INIT_ARGS', '')}")

    logging.info(f"NODE:{node_name} EVENT:INIT_START")
    local_devices = jax.local_devices()
    logging.info(
        f"Process {jax.process_index()} on {node_name} | Local Devices: {len(local_devices)}"
    )

    logging.info(f"NODE:{node_name} EVENT:WARMUP_START")
    A_stack, W1_stack, W2_stack = init_payloads(jax.random.PRNGKey(RANDOM_SEED))

    warmup_res = hardware_health_stress(
        A_stack, W1_stack, W2_stack, NUM_ITERATIONS_WARMUP
    )
    jax.block_until_ready(warmup_res)

    logging.info(f"NODE:{node_name} EVENT:MEASURE_START")
    start_time = time.time()
    outputs = hardware_health_stress(
        A_stack, W1_stack, W2_stack, NUM_ITERATIONS_MEASURE
    )
    jax.block_until_ready(outputs)
    duration = time.time() - start_time

    flops_per_matmul = 2.0 * (MATRIX_DIM**3)
    total_flops_per_loop = flops_per_matmul * (NUM_MATMULS_PER_ITERATION * 2)
    total_flops = total_flops_per_loop * NUM_ITERATIONS_MEASURE
    tflops_per_core = (total_flops / duration) / (1e12)

    if experimental_enabled():
        results_valid = stress_other_units_and_check_results(
            node_name, local_devices[0].device_kind, A_stack, W1_stack, outputs
        )
    else:
        results_valid = matmul_results_match(outputs)
    report_results(node_name, results_valid, duration, tflops_per_core)


if __name__ == "__main__":
    main()
