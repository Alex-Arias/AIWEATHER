"""
GraphCast operational runner.
"""

from __future__ import annotations

from .px_runner import PXRunner


class GraphCastRunner(PXRunner):

    MODEL_NAME = "graphcast"