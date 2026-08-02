"""
GraphCast operational runner.
"""

from __future__ import annotations

from earth2studio.models.px import GraphCastOperational
from earth2studio.run import deterministic
from earth2studio.data import GFS

from .base_runner import BaseRunner


class GraphCastRunner(BaseRunner):

    # ---------------------------------------------------------
    # Model
    # ---------------------------------------------------------

    def load_model(self):

        if self.model is not None:
            return

        print()
        print("Loading GraphCastOperational...")

        package = GraphCastOperational.load_default_package()

        self.model = GraphCastOperational.load_model(package)

        print("GraphCast loaded.")

    # ---------------------------------------------------------
    # Data
    # ---------------------------------------------------------

    def load_data(self):

        if self.data is not None:
            return

        print()
        print("Initializing GFS...")

        self.data = GFS()

        print("GFS initialized.")

    # ---------------------------------------------------------
    # Forecast
    # ---------------------------------------------------------

    def run_forecast(self, request):

        print()
        print("Running deterministic forecast...")

        deterministic(
            time=[request.init_time],
            nsteps=request.lead_time // 6,
            prognostic=self.model,
            data=self.data,
            io=self.io,
        )

        print()
        print("Forecast completed.")