from chs.plugins.tpu_collectives.renderer import TpuCollectivesRenderer
from rich.text import Text


class TpuCollectivesSingleNodeRenderer(TpuCollectivesRenderer):
    def _get_summary_title(self) -> Text:
        return Text(
            "\nTPU Collectives Single Node Performance Summary",
            style="bold underline cyan",
        )
