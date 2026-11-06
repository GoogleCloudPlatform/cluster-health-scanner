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

from concurrent.futures import ThreadPoolExecutor
import functools
import logging
import threading
from typing import Any
from chs.resources import normalize_machine_type
from chs.exceptions import ChsSchedulerException

logger = logging.getLogger(__name__)

_BATCH_SIZE = 50
_MAX_CONCURRENT_BATCHES = 8


@functools.cache
def get_default_credentials() -> tuple[Any, str | None]:
    """Returns the process-wide Application Default Credentials and project.

    Raises:
        google.auth.exceptions.DefaultCredentialsError: If no credentials are
            configured.
    """
    import google.auth

    return google.auth.default()


def query_gce_instances_info(
    project: str, nodes: list[str]
) -> dict[str, dict[str, Any]]:
    """Queries GCE API for instance information.

    Args:
        project: GCP project ID.
        nodes: List of GCE instance names.

    Returns:
        A dict mapping instance name to a dict containing 'machine_type',
        'instance_id' (string representation of numeric ID), and optional
        'block' and 'subblock' fields when present.
    """
    # Lazy import to avoid heavy compute_v1 import overhead on CLI startup.
    from google.api_core.exceptions import GoogleAPICallError
    from google.auth.exceptions import GoogleAuthError
    from google.cloud import compute_v1

    # The client talks REST through a requests.Session, which isn't
    # thread-safe, so each worker thread creates its own client.
    local = threading.local()

    def thread_client() -> Any:
        if not hasattr(local, "client"):
            local.client = compute_v1.InstancesClient()
        return local.client

    def query_batch(batch: list[str]) -> dict[str, dict[str, Any]]:
        filter_str = " OR ".join([f'name = "{name}"' for name in batch])
        batch_info = {}
        try:
            request = compute_v1.AggregatedListInstancesRequest(
                project=project,
                filter=filter_str,
            )
            page_result = thread_client().aggregated_list(request=request)

            for zone, instances_in_zone in page_result:
                if not instances_in_zone.instances:
                    continue
                for instance in instances_in_zone.instances:
                    if instance.name in batch:
                        normalized_machine_type = normalize_machine_type(
                            instance.machine_type
                        )
                        instance_info = {
                            "machine_type": normalized_machine_type,
                            "instance_id": str(instance.id),
                            "zone": zone.removeprefix("zones/"),
                        }
                        if instance.resource_status:
                            topo = (
                                instance.resource_status.physical_host_topology
                            )
                            if topo:
                                if topo.block and isinstance(topo.block, str):
                                    instance_info["block"] = topo.block
                                if topo.subblock and isinstance(
                                    topo.subblock, str
                                ):
                                    instance_info["subblock"] = topo.subblock
                        batch_info[instance.name] = instance_info
        except (GoogleAPICallError, GoogleAuthError) as e:
            raise ChsSchedulerException(f"Failed to query GCE API: {e}") from e
        return batch_info

    # Batch requests in chunks of 50 to avoid URL length limits, and send a
    # few batches at a time.
    batches = [
        nodes[i : i + _BATCH_SIZE] for i in range(0, len(nodes), _BATCH_SIZE)
    ]
    if not batches:
        return {}
    info_map = {}
    executor = ThreadPoolExecutor(
        max_workers=min(len(batches), _MAX_CONCURRENT_BATCHES)
    )
    try:
        for batch_info in executor.map(query_batch, batches):
            info_map.update(batch_info)
    finally:
        executor.shutdown(wait=True, cancel_futures=True)
    return info_map
