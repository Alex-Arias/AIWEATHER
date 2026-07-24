"""
GraphCast model adapter.
"""

from __future__ import annotations

from aiweather.forecast import Forecast

from .base_model import BaseModel


class GraphCastModel(BaseModel):
    """
    AIWeather adapter for GraphCast.
    """

    name = "graphcast"

    def load(self) -> None:
        """
        Load the GraphCast model.

        Actual Earth2Studio loading will be implemented
        in the next milestone.
        """
        self._loaded = True

    def predict(self, dataset):
        """
        Run a forecast.

        Placeholder implementation.
        """
        raise NotImplementedError(
            "GraphCast inference not implemented yet."
        )

    def unload(self) -> None:
        """
        Release model resources.
        """
        self._loaded = False