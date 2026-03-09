"""Entry point for the Cluster Health Scanner (CHS) command-line interface.

This script initializes logging and delegates argument parsing and command
execution to the CHS application library.
"""

import logging
import sys

from chs import cli


def main():
    logging.basicConfig(
        level=logging.INFO,
        stream=sys.stderr,
        format="%(asctime)s %(levelname)-8s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    cli.main()


if __name__ == "__main__":
    main()
