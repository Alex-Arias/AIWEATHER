"""
Generic runner for Earth2Studio prognostic (PX) models.
"""

from __future__ import annotations

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

        from aiweather.backends.earth2studio import (
            load_px_model,
        )

        self._check_model_name()

        print()
        print(f"Loading {self.MODEL_NAME}...")

        self.model = load_px_model(
            self.MODEL_NAME
        )

        print(
            f"{self.MODEL_NAME} loaded."
        )


    def load_data(
        self,
        request,
    ):

        from aiweather.backends.earth2studio import (
            load_data_source,
        )

        self._check_model_name()

        print()
        print(
            f"Initializing datasource for "
            f"{self.MODEL_NAME}..."
        )

        self.data = load_data_source(
            self.MODEL_NAME,
            datasource=request.datasource,
            source=request.datasource_source,
        )

        print(
            "Datasource initialized."
        )

    def run_forecast(self, request):

        from aiweather.backends.earth2studio.inference import (
            run_forecast,
        )

        print()
        print(
            "Running deterministic forecast..."
        )

        if (
            request.lead_time
            % self.MODEL_TIMESTEP
            != 0
        ):
            raise ValueError(
                f"{self.MODEL_NAME} requires lead_time "
                f"to be divisible by "
                f"{self.MODEL_TIMESTEP} hours."
            )

        run_forecast(
            time=request.init_time,
            nsteps=(
                request.lead_time
                // self.MODEL_TIMESTEP
            ),
            prognostic=self.model,
            data=self.data,
            io=self.io,
        )

        OutputManager.write_forecast_metadata(
            request.output_path,
            request,
        )

        print()
        print(
            "Forecast completed."
        )