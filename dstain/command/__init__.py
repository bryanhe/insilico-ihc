"""Command line entry points."""

from .analyze import analyze
from .baseline import baseline
from .evaluate import evaluate
from .ihc import ihc
from .process import process
from .register import register
from .train import train
from .visualize import visualize

__all__ = ["analyze", "baseline", "evaluate", "ihc", "process", "register",
           "stain", "train", "ui", "visualize"]
