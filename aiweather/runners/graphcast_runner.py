"""
GraphCast operational runner.
"""

from __future__ import annotations

from .base_runner import BaseRunner


class GraphCastRunner(BaseRunner):
    """
    Execute GraphCast forecasts using Earth2Studio.
    """

    def run(
        self,
        request,
    ):
        raise NotImplementedError(
            "GraphCastRunner not implemented yet."
        )