from dstain.__version__ import __version__
from dstain.config import config
import dstain.constant as constant
import dstain.command as command
import dstain.datasets as datasets
import dstain.transforms as transforms
import dstain.utils as utils

import click


@click.group()
def main():
    pass


del click


main.add_command(command.analyze)
main.add_command(command.baseline)
main.add_command(command.evaluate)
main.add_command(command.ihc)
main.add_command(command.process)
main.add_command(command.register)
main.add_command(command.train)
main.add_command(command.visualize)

__all__ = [
    "__version__", "blocks", "command", "config", "constant", "datasets",
    "main", "transforms", "utils"
]
