import importlib.resources as pkg_resources
import logging
import os
import shutil

import chs
from chs.schedulers.base import BaseScheduler

logger = logging.getLogger(__name__)


class DefaultScheduler(BaseScheduler):
  """Fallback scheduler for non-specific schedulers (e.g.

  bare metal default).
  """

  def __init__(self, name: str):
    self._name = name

  @property
  def name(self) -> str:
    return self._name

  def get_nodes_under_test(self) -> list[str]:
    return []

  def write_ramble_config(
      self,
      workspace_config_dir: str,
      suite: str,
      tests: list[str] | None,
      machine_type: str,
  ) -> None:
    with pkg_resources.path(chs, "ramble.yaml") as p:
      shutil.copy(p, os.path.join(workspace_config_dir, "ramble.yaml"))
      logger.debug("Copied default ramble.yaml to %s", workspace_config_dir)
