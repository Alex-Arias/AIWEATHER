"""
GraphCast operational runner.
"""

from __future__ import annotations

from earth2studio.models.px import GraphCastOperational

from .base_runner import BaseRunner


class GraphCastRunner(BaseRunner):
    """
    Execute GraphCast forecasts using Earth2Studio.
    """

    def __init__(self):

        self.model = None

    # ---------------------------------------------------------
    # Model
    # ---------------------------------------------------------

    def load_model(self):

        print()

        print("Loading GraphCastOperational...")

        package = GraphCastOperational.load_default_package()

        self.model = GraphCastOperational.load_model(package)

        print("GraphCast loaded.")

    # ---------------------------------------------------------
    # Data
    # ---------------------------------------------------------

    def load_data(self):

        raise NotImplementedError

    # ---------------------------------------------------------
    # Forecast
    # ---------------------------------------------------------

    def run(
        self,
        request,
    ):

        if self.model is None:

            self.load_model()

        raise NotImplementedError