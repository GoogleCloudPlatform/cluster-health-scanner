"""Methods for health check metrics.
"""

from typing import Any, Dict


def log_dict(
    test_name: str,
    did_pass: bool,
    node_name: str,
    result_data: Dict[str, Any],
) -> Dict[str, Any]:
  return {
      "test_name": test_name,
      "did_pass": did_pass,
      "node_name": node_name,
      "result_data": result_data,
  }
