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
    " --megascale_error_reporter_abort_on_hang=true"
)
os.environ["LIBTPU_INIT_ARGS"] = (
    os.environ.get("LIBTPU_INIT_ARGS", "") + init_args
)

import numpy as np
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
NUM_ITERATIONS_WARMUP = 100
NUM_BURN_ROUND_TRIPS = 7500
NUM_ITERATIONS_MEASURE = 2 * NUM_BURN_ROUND_TRIPS
WALK_CHUNK_BYTES = 1 << 30
WALK_CHUNK_WORDS = WALK_CHUNK_BYTES // jnp.dtype(jnp.uint32).itemsize
WALK_VERIFY_TEMP_CHUNKS = 2
WALK_PEAK_FRACTION_OF_LIMIT = 0.90
DATA_SEED = 42
BURN_DATA_STREAM = 0
WALK_FIRST_STREAM = BURN_DATA_STREAM + 1
DEVICE_AXIS = "device"
REPLICATED = jax.sharding.PartitionSpec()
PER_DEVICE = jax.sharding.PartitionSpec(DEVICE_AXIS)


def _local_mesh():
    return jax.sharding.Mesh(
        np.array(jax.local_devices()), axis_names=DEVICE_AXIS
    )


def _random_words(stream, num_words):
    key = jax.random.fold_in(jax.random.key(DATA_SEED), stream)
    return jax.random.bits(key, (num_words,), dtype=jnp.uint32)


@functools.partial(jax.jit, static_argnums=(1,))
def random_words_on_every_device(stream, num_words):
    return jax.shard_map(
        lambda local_stream: _random_words(local_stream, num_words),
        mesh=_local_mesh(),
        in_specs=REPLICATED,
        out_specs=REPLICATED,
    )(stream)


@jax.jit
def _init_single_memory(key):
    return jax.random.randint(
        key, (NUM_ELEMENTS,), minval=0, maxval=2147483647, dtype=jnp.uint32
    )


def init_memory(key):
    local_devices = jax.local_devices()
    sharding = jax.sharding.NamedSharding(_local_mesh(), REPLICATED)
    with jax.default_device(local_devices[0]):
        a0 = _init_single_memory(key)
        a_np = jax.device_get(a0)

    shards = [a0] + [jax.device_put(a_np, d) for d in local_devices[1:]]
    A = jax.make_array_from_single_device_arrays(
        (NUM_ELEMENTS,), sharding, shards
    )
    jax.block_until_ready(A)
    return A


def init_regenerable_memory():
    A = random_words_on_every_device(BURN_DATA_STREAM, NUM_ELEMENTS)
    jax.block_until_ready(A)
    return A


def _hbm_burn_step(x):
    return jax.lax.bitwise_not(x)


@functools.partial(jax.jit, static_argnums=(1,))
def _hbm_burn_device_loop(a, num_iters):
    def _loop(x):
        return jax.lax.fori_loop(
            0, num_iters, lambda _, v: _hbm_burn_step(v), x
        )

    return jax.shard_map(
        _loop,
        mesh=_local_mesh(),
        in_specs=REPLICATED,
        out_specs=REPLICATED,
    )(a)


def hbm_burn(a, num_iters):
    remaining = int(num_iters)
    x = a
    while remaining > 0:
        step = min(NUM_ITERATIONS_WARMUP, remaining)
        x = _hbm_burn_device_loop(x, step)
        remaining -= step
    return x


@jax.jit
def _hbm_checksum_single(a):
    a_int64 = a.astype(jnp.int64)
    return jnp.sum(a_int64 ^ jnp.roll(a_int64, shift=1), dtype=jnp.int64)


def hbm_checksum(a):
    return [_hbm_checksum_single(shard.data) for shard in a.addressable_shards]


def checksums_match_device_0(burned):
    checksums = jax.device_get(hbm_checksum(burned))
    return all(checksum == checksums[0] for checksum in checksums)


@functools.partial(jax.jit, static_argnames=("inverted",))
def count_mismatches_per_device(buf, stream, inverted):
    def _count(local_buf, local_stream):
        expected = _random_words(local_stream, local_buf.shape[0])
        if inverted:
            expected = jax.lax.bitwise_not(expected)
        return jnp.count_nonzero(local_buf != expected, keepdims=True)

    return jax.shard_map(
        _count,
        mesh=_local_mesh(),
        in_specs=(REPLICATED, REPLICATED),
        out_specs=PER_DEVICE,
    )(buf, stream)


@functools.partial(jax.jit, donate_argnums=(0,))
def _invert_in_place(buf):
    return jax.lax.bitwise_not(buf)


def _total(per_device_counts):
    return int(np.sum(jax.device_get(per_device_counts)))


def count_mismatches_both_polarities(buffers, streams):
    counts = [
        count_mismatches_per_device(buf, stream, inverted=False)
        for buf, stream in zip(buffers, streams)
    ]
    mismatches = _total(counts)
    inverted = [_invert_in_place(buf) for buf in buffers]
    counts = [
        count_mismatches_per_device(buf, stream, inverted=True)
        for buf, stream in zip(inverted, streams)
    ]
    return mismatches + _total(counts)


def walk_chunk_count():
    peak_chunks = (
        int(_hbm_bytes * WALK_PEAK_FRACTION_OF_LIMIT) // WALK_CHUNK_BYTES
    )
    return peak_chunks - WALK_VERIFY_TEMP_CHUNKS


def capacity_walk():
    streams = range(WALK_FIRST_STREAM, WALK_FIRST_STREAM + walk_chunk_count())
    chunks = [
        random_words_on_every_device(stream, WALK_CHUNK_WORDS)
        for stream in streams
    ]
    return count_mismatches_both_polarities(chunks, streams)


def experimental_enabled():
    return os.environ.get("CHS_EXPERIMENTAL") == "true"


def main():
    experimental = experimental_enabled()
    logging.info("Starting TPU HBM stress script...")
    logging.info(f"ACCELERATOR_TYPE: {accel_type}, Deduced P_STATE: {p_state}")
    logging.info(
        f"NUM_ELEMENTS={NUM_ELEMENTS} NUM_ITERATIONS_MEASURE={NUM_ITERATIONS_MEASURE}"
    )
    if experimental:
        logging.info(
            f"HBM_BYTES_LIMIT={_hbm_bytes} WALK_CHUNKS={walk_chunk_count()}"
        )
    local_devices = jax.local_devices()
    node_name = os.environ.get("MY_NODE_NAME", "UnknownHost")

    logging.info(f"NODE:{node_name} EVENT:INIT_START")
    logging.info(
        f"Process {jax.process_index()} on {node_name} | Local Devices: {len(local_devices)}"
    )

    logging.info(f"NODE:{node_name} EVENT:WARMUP_START")
    if experimental:
        A_mem = init_regenerable_memory()
    else:
        A_mem = init_memory(jax.random.PRNGKey(DATA_SEED))

    jax.block_until_ready(hbm_burn(A_mem, NUM_ITERATIONS_WARMUP))

    logging.info(f"NODE:{node_name} EVENT:MEASURE_START")
    start_time = time.time()

    outputs = hbm_burn(A_mem, NUM_ITERATIONS_MEASURE)
    jax.block_until_ready(outputs)

    duration = time.time() - start_time
    del A_mem

    bytes_per_iteration = (NUM_ELEMENTS * 4) * 2
    total_bytes_transferred = bytes_per_iteration * NUM_ITERATIONS_MEASURE
    gb_per_second = (total_bytes_transferred / duration) / (1e9)

    devices_per_chip = max(1, len(local_devices) // 4)
    chip_bandwidth = gb_per_second * devices_per_chip

    if experimental:
        burn_mismatches = count_mismatches_both_polarities(
            [outputs], [BURN_DATA_STREAM]
        )
        del outputs
        walk_mismatches = capacity_walk()
        logging.info(
            f"NODE:{node_name} EVENT:HBM_MISMATCHES BURN:{burn_mismatches} WALK:{walk_mismatches}"
        )
        sdc_detected = bool(burn_mismatches or walk_mismatches)
    else:
        sdc_detected = not checksums_match_device_0(outputs)

    if sdc_detected:
        logging.error(
            f"NODE:{node_name} EVENT:STRESS_FAIL. TPU devices failed SDC check"
        )
    else:
        logging.info(
            f"NODE:{node_name} EVENT:MEASURE_END DURATION:{duration:.2f}s BANDWIDTH_GBPS:{chip_bandwidth:.2f}"
        )
        logging.info(f"NODE:{node_name} EVENT:STRESS_PASS")


if __name__ == "__main__":
    main()
