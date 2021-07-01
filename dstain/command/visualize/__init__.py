import click

from .samples import samples
from .register import register


@click.group()
def visualize():
    pass


visualize.add_command(samples)
visualize.add_command(register)

__all__ = ["samples", "register"]
