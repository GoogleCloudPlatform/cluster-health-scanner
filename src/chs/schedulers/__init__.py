"""Module for managing cluster schedulers."""

from chs.schedulers.base import BaseScheduler
from chs.schedulers.default import DefaultScheduler
from chs.schedulers.slurm import SlurmScheduler
from chs.schedulers.gke import GKEScheduler


def get_scheduler(scheduler: str, **kwargs) -> BaseScheduler:
    """Factory to get the right scheduler instance."""
    app_data = kwargs.get("app_data")
    if scheduler == "slurm":
        return SlurmScheduler(
            nodes=kwargs.get("nodes"),
            nodelist=kwargs.get("nodelist"),
            partition=kwargs.get("partition"),
            app_data=app_data,
        )
    if scheduler == "gke":
        return GKEScheduler(
            nodes=kwargs.get("nodes"),
            nodepool=kwargs.get("nodepool"),
            app_data=app_data,
        )
    return DefaultScheduler(scheduler, app_data=app_data)
