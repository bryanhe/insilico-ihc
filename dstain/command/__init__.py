"""Command line entry points."""

from .analyze import analyze
from .annotate import annotate
from .baseline import baseline
from .deepzoom import deepzoom
from .evaluate import evaluate
from .example import example
from .ihc import ihc
from .oracle import oracle
from .process import process
from .register import register
from .stain import stain
from .train import train
from .ui import ui
from .visualize import visualize

__all__ = ["analyze", "annotate", "baseline", "deepzoom", "evaluate", "example", "ihc",
           "oracle", "process", "register", "stain", "train", "ui", "visualize"]
