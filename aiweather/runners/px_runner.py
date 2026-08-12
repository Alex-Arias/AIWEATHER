"""
Generic runner for Earth2Studio prognostic (PX) models.
"""

from __future__ import annotations

from earth2studio.io import ZarrBackend

from aiweather.backends.earth2studio import (
    load_px_model,
    load_data_source,
)

from aiweather.backends.earth2studio.inference import (
    run_forecast,
)

from aiweather.output import OutputManager

from .base_runner import BaseRunner


class PXRunner(BaseRunner):

    MODEL_NAME = None

    # GraphCast native timestep (hours)
    MODEL_TIMESTEP = 6

    def __init__(self):
        super().__init__()

    def _check_model_name(self):

        if self.MODEL_NAME is None:
            raise ValueError(
                "MODEL_NAME must be defined by subclasses."
            )

    def load_model(self):

        self._check_model_name()

        print()
        print(f"Loading {self.MODEL_NAME}...")

        self.model = load_px_model(self.MODEL_NAME)

        print(f"{self.MODEL_NAME} loaded.")

    def load_data(self):

        self._check_model_name()

        print()
        print(f"Initializing datasource for {self.MODEL_NAME}...")

        self.data = load_data_source(self.MODEL_NAME)

        print("Datasource initialized.")


    def run_forecast(self, request):

        print()
        print("Running deterministic forecast...")

        run_forecast(
            time=request.init_time,
            nsteps=request.lead_time // self.MODEL_TIMESTEP,
            prognostic=self.model,
            data=self.data,
            io=self.io,
        )

        OutputManager.write_forecast_metadata(
            request.output_path,
            request,
        )

        print()
        print("Forecast completed.")
