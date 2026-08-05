"""
Generic runner for Earth2Studio prognostic (PX) models.
"""

from __future__ import annotations

from aiweather.backends.earth2studio import (
    load_px_model,
    load_data_source,
)

from aiweather.backends.earth2studio.inference import (
    run_forecast,
)

from .base_runner import BaseRunner


class PXRunner(BaseRunner):
    """
    Generic runner for Earth2Studio prognostic models.

    Concrete subclasses only need to define MODEL_NAME.
    """

    MODEL_NAME = None

    def __init__(self):
        super().__init__()

    # ---------------------------------------------------------
    # Validation
    # ---------------------------------------------------------

    def _check_model_name(self):
        if self.MODEL_NAME is None:
            raise ValueError(
                "MODEL_NAME must be defined by subclasses."
            )

    # ---------------------------------------------------------
    # Model
    # ---------------------------------------------------------

    def load_model(self):

        self._check_model_name()

        print()
        print(f"Loading {self.MODEL_NAME}...")

        self.model = load_px_model(self.MODEL_NAME)

        print(f"{self.MODEL_NAME} loaded.")

    # ---------------------------------------------------------
    # Data
    # ---------------------------------------------------------

    def load_data(self):

        self._check_model_name()

        print()
        print(f"Initializing datasource for {self.MODEL_NAME}...")

        self.data = load_data_source(self.MODEL_NAME)

        print("Datasource initialized.")

    # ---------------------------------------------------------
    # Forecast
    # ---------------------------------------------------------

    def run_forecast(self, request):

        print()
        print("Running deterministic forecast...")

        run_forecast(
            time=[request.init_time],
            nsteps=request.lead_time // 6,
            prognostic=self.model,
            data=self.data,
            io=self.io,
        )

        print()
        print("Forecast completed.")