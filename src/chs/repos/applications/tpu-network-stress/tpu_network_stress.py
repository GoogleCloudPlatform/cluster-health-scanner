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
import collections

init_args = (
    " --tpu_link_up_check_timeout=15s"
    " --megascale_error_reporter_abort_on_hang=true"
)
os.environ["LIBTPU_INIT_ARGS"] = (
    os.environ.get("LIBTPU_INIT_ARGS", "") + init_args
)
import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec
from jax.experimental import multihost_utils
import numpy as np

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

PAYLOAD_SIZE = int(os.environ.get("PAYLOAD_SIZE_MB", 1024)) * 1024 * 1024
NUM_ITERATIONS_WARMUP = 5
# MUST be an even number to ensure the payload remains the same after the pair-wise transfer bounces.
NUM_ITERATIONS_MEASURE = 1000


def calculate_bandwidth(payload_nbytes, num_cores_per_chip, duration):
    total_bytes_per_link = NUM_ITERATIONS_MEASURE * (
        payload_nbytes * num_cores_per_chip * 2
    )
    return (total_bytes_per_link / duration) / (1024**3)


def build_permutation_pairs(axis_size, is_odd_iteration, mesh_shape):
    # Wrap-around only exists if all dimensions in the topology are multiples of 4
    has_wrap_around = all(s % 4 == 0 for s in mesh_shape)

    pairs = []
    for i in range(axis_size):
        if is_odd_iteration:
            if i % 2 != 0:
                target = i + 1
                if target < axis_size:
                    pairs.extend([(i, target), (target, i)])
                elif has_wrap_around and target == axis_size:
                    pairs.extend([(i, 0), (0, i)])
        else:
            if i % 2 == 0:
                target = i + 1
                if target < axis_size:
                    pairs.extend([(i, target), (target, i)])
    return pairs


@functools.lru_cache(maxsize=None)
def get_jitted_swap(perm_list, active_axis, global_mesh):
    @functools.partial(
        jax.jit,
        in_shardings=(NamedSharding(global_mesh, PartitionSpec()), None),
        out_shardings=NamedSharding(global_mesh, PartitionSpec()),
    )
    def _swap(local_payload, num_iterations):
        def _shard_map_swap(val):
            val = jax.lax.pcast(val, (active_axis,), to="varying")

            def body_fn(i, v):
                swapped = jax.lax.ppermute(
                    v, axis_name=active_axis, perm=perm_list
                )
                return swapped

            return jax.lax.fori_loop(0, num_iterations, body_fn, val)

        mapped_fn = jax.shard_map(
            _shard_map_swap,
            mesh=global_mesh,
            in_specs=PartitionSpec(),
            out_specs=PartitionSpec(),
            check_vma=False,
        )
        return mapped_fn(local_payload)

    return _swap


def run_swap(payload, perm_list, active_axis, num_iterations, global_mesh):
    jitted_swap = get_jitted_swap(tuple(perm_list), active_axis, global_mesh)
    return jitted_swap(payload, jnp.int32(num_iterations))


def test_permutation_group(payload, perm_list, axis_name, global_mesh):
    axis_idx = {"x": 0, "y": 1, "z": 2}[axis_name]
    active_in_group = False
    for d in jax.local_devices():
        coord_val = d.coords[axis_idx]
        if any(src == coord_val for src, dst in perm_list):
            active_in_group = True
            break

    warmup = run_swap(
        payload, perm_list, axis_name, NUM_ITERATIONS_WARMUP, global_mesh
    )
    warmup.block_until_ready()

    barrier_name = f"barrier_{axis_name}_{hash(tuple(perm_list))}"
    original_payload = payload

    multihost_utils.sync_global_devices(barrier_name)

    start_time = time.time()
    payload = run_swap(
        payload,
        perm_list,
        axis_name,
        NUM_ITERATIONS_MEASURE,
        global_mesh,
    )
    payload.block_until_ready()
    end_time = time.time()

    if not active_in_group:
        return None, True

    duration = end_time - start_time

    is_data_valid = True
    orig_shards = {
        s.device.id: s.data for s in original_payload.addressable_shards
    }

    for shard in payload.addressable_shards:
        d = shard.device
        # Check if THIS specific device was active in the permutation
        coord_val = d.coords[axis_idx]
        device_active = any(src == coord_val for src, dst in perm_list)
        if not device_active:
            continue  # Idle devices return zeros from ppermute, skipping validation

        local_payload = np.asarray(shard.data)
        local_orig = np.asarray(orig_shards[d.id])

        if not np.array_equal(local_payload, local_orig):
            diff_bytes = np.sum(local_payload != local_orig)
            logging.error(
                f"Hardware SDC confirmed on device {d.id} (coords {d.coords}): {diff_bytes} bytes corrupted out of {local_payload.size}"
            )
            is_data_valid = False

    if not is_data_valid:
        logging.error(f"Silent data corruption detected on axis {axis_name}!")

    # Since 'core' dimension is not permuted, both cores on the chip send/receive simultaneously over the same cable.
    core_multiplier = global_mesh.shape["core"]
    bw = calculate_bandwidth(payload.nbytes * core_multiplier, 1, duration)
    return bw, is_data_valid


def main():
    logging.info("Starting TPU ICI stress script...")
    logging.info(
        f"PAYLOAD_SIZE={PAYLOAD_SIZE} NUM_ITERATIONS_MEASURE={NUM_ITERATIONS_MEASURE}"
    )
    jax.distributed.initialize()
    devices = jax.devices()
    local_devices = jax.local_devices()

    node_name = os.environ.get("MY_NODE_NAME", "")

    if not devices:
        logging.error(
            "XlaRuntimeError: No TPU devices found by JAX. Please check the TPU driver and hardware."
        )
        raise ValueError("No TPU devices found by JAX")

    logging.info(f"NODE:{node_name} EVENT:INIT_START")
    logging.info(
        f"Process {jax.process_index()} on {node_name} | Local Devices: {len(local_devices)} | Total Devices: {len(devices)}"
    )

    shape_x = max(d.coords[0] for d in devices) + 1
    shape_y = max(d.coords[1] for d in devices) + 1
    shape_z = max(d.coords[2] for d in devices) + 1

    max_cores_per_chip = len(
        [d for d in devices if tuple(d.coords) == (0, 0, 0)]
    )

    devices_4d = np.empty(
        (shape_x, shape_y, shape_z, max_cores_per_chip), dtype=object
    )
    current_core = collections.defaultdict(int)
    for d in devices:
        coord = tuple(d.coords)
        c_idx = current_core[coord]
        devices_4d[d.coords[0], d.coords[1], d.coords[2], c_idx] = d
        current_core[coord] += 1

    logging.info(
        f"Calculated shape: x={shape_x}, y={shape_y}, z={shape_z}, cores_per_chip={max_cores_per_chip}"
    )

    if None in devices_4d.flat:
        logging.error(
            "Invalid topology detected! Not all coordinates form a dense 4D grid."
        )
        return

    global_mesh = Mesh(devices_4d, axis_names=("x", "y", "z", "core"))
    sharding = NamedSharding(global_mesh, PartitionSpec())
    global_seed = 42
    local_data_list = []

    for d in local_devices:
        unique_device_seed = global_seed + d.id
        key = jax.random.PRNGKey(unique_device_seed)

        # Payload Tiling: Generate a small 1MB high-entropy block, then tile it.
        # This avoids XLA PRNG state-splitting OOMs during compilation.
        chunk_size = 1024 * 1024
        small_entropy = jax.random.randint(
            key, (chunk_size,), minval=-128, maxval=127, dtype=jnp.int8
        )
        repeats = max(1, PAYLOAD_SIZE // chunk_size)
        device_data = jnp.tile(small_entropy, repeats)
        local_data_list.append(jax.device_put(device_data, d))

    payload = jax.make_array_from_single_device_arrays(
        local_data_list[0].shape, sharding, local_data_list
    )
    payload.block_until_ready()

    all_results = []
    topology_shape = (shape_x, shape_y, shape_z)

    for axis_idx, axis_name in enumerate(["x", "y", "z"]):
        axis_shape = topology_shape[axis_idx]
        if axis_shape <= 1:
            continue

        axis_bandwidths = []
        axis_valid = True

        for is_odd in [False, True]:
            iter_name = f"{axis_name}_{'odd' if is_odd else 'even'}"
            perm_list = build_permutation_pairs(
                axis_shape, is_odd, topology_shape
            )

            if not perm_list:
                continue

            logging.info(f"=== Testing Group {iter_name} ===")
            bandwidth, is_data_valid = test_permutation_group(
                payload, perm_list, axis_name, global_mesh
            )

            if bandwidth is None:
                logging.info(
                    f"[{iter_name}] Group SKIPPED: No local devices in permutation"
                )
                continue

            if not is_data_valid:
                logging.error(
                    f"[{iter_name}] Group FAILED: SDC (Bit flips) Detected"
                )

            else:
                logging.info(
                    f"[{iter_name}] Group Bandwidth: {bandwidth:.2f} GB/s"
                )

            axis_bandwidths.append(bandwidth)
            axis_valid = axis_valid and is_data_valid

        if not axis_bandwidths:
            continue

        final_bandwidth = min(axis_bandwidths)
        all_results.append((node_name, axis_name, final_bandwidth, axis_valid))

    logging.info("=== SUMMARY OF LOCAL HOST ICI BANDWIDTH ===")
    for host, axis, bw, is_valid in all_results:
        if not is_valid:
            logging.error(
                f"Host: {host} | Axis: {axis.upper()} | {bw:.2f} GB/s | FAILED: SDC (Bit flips) Detected"
            )
        else:
            logging.info(f"Host: {host} | Axis: {axis.upper()} | {bw:.2f} GB/s")
    logging.info("Waiting for all hosts to finish before exiting...")
    multihost_utils.sync_global_devices("end_of_script")
    logging.info("All hosts finished. Exiting cleanly.")


if __name__ == "__main__":
    main()
