"""
Generic runner for Earth2Studio prognostic (px) models.
"""

from __future__ import annotations

from earth2studio.run import deterministic

from aiweather.registry.models import MODEL_REGISTRY
from aiweather.registry.datasources import DATA_REGISTRY

from .base_runner import BaseRunner


class PXRunner(BaseRunner):

    MODEL_NAME = None

    def __init__(self):
        super().__init__()

    # ---------------------------------------------------------
    # Model
    # ---------------------------------------------------------

    def load_model(self):

        if self.MODEL_NAME is None:
            raise ValueError(
                "MODEL_NAME must be defined by subclasses."
            )

        print()
        print(f"Loading {self.MODEL_NAME}...")

        ModelClass = MODEL_REGISTRY[self.MODEL_NAME]

        package = ModelClass.load_default_package()

        self.model = ModelClass.load_model(package)

        print(f"{self.MODEL_NAME} loaded.")

    # ---------------------------------------------------------
    # Data
    # ---------------------------------------------------------

    def load_data(self):

        if self.MODEL_NAME is None:
            raise ValueError(
                "MODEL_NAME must be defined by subclasses."
            )

        print()

        DataClass = DATA_REGISTRY[self.MODEL_NAME]

        print(f"Initializing {DataClass.__name__}...")

        self.data = DataClass()

        print(f"{DataClass.__name__} initialized.")

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