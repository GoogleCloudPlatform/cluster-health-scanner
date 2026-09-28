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

import logging
from typing import Any
from chs.resources import normalize_machine_type
from chs.exceptions import ChsSchedulerException

logger = logging.getLogger(__name__)


def query_gce_instances_info(
    project: str, nodes: list[str]
) -> dict[str, dict[str, Any]]:
    """Queries GCE API for instance information.

    Args:
        project: GCP project ID.
        nodes: List of GCE instance names.

    Returns:
        A dict mapping instance name to a dict containing 'machine_type'
        and 'instance_id' (string representation of numeric ID).
    """
    # Lazy import to avoid heavy compute_v1 import overhead on CLI startup.
    from google.api_core.exceptions import GoogleAPICallError
    from google.auth.exceptions import GoogleAuthError
    from google.cloud import compute_v1

    try:
        client = compute_v1.InstancesClient()
    except (GoogleAPICallError, GoogleAuthError) as e:
        raise ChsSchedulerException(f"Failed to query GCE API: {e}") from e

    info_map = {}

    # Batch requests in chunks of 50 to avoid URL length limits
    for i in range(0, len(nodes), 50):
        batch = nodes[i : i + 50]
        filter_str = " OR ".join([f'name = "{name}"' for name in batch])
        try:
            request = compute_v1.AggregatedListInstancesRequest(
                project=project,
                filter=filter_str,
            )
            page_result = client.aggregated_list(request=request)

            for zone, instances_in_zone in page_result:
                if not instances_in_zone.instances:
                    continue
                for instance in instances_in_zone.instances:
                    if instance.name in batch:
                        normalized_machine_type = normalize_machine_type(
                            instance.machine_type
                        )
                        info_map[instance.name] = {
                            "machine_type": normalized_machine_type,
                            "instance_id": str(instance.id),
                        }
        except (GoogleAPICallError, GoogleAuthError) as e:
            raise ChsSchedulerException(f"Failed to query GCE API: {e}") from e

    return info_map
