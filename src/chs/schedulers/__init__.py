"""Module for managing cluster schedulers."""

from chs.schedulers.base import BaseScheduler
from chs.schedulers.default import DefaultScheduler
from chs.schedulers.slurm import SlurmScheduler


def get_scheduler(scheduler: str, **kwargs) -> BaseScheduler:
    """Factory to get the right scheduler instance."""
    if scheduler == "slurm":
        return SlurmScheduler(
            nodes=kwargs.get("nodes"),
            nodelist=kwargs.get("nodelist"),
            partition=kwargs.get("partition"),
        )
    return DefaultScheduler(scheduler)
