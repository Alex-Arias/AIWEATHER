"""
Forecast object for AIWeather.
"""

from __future__ import annotations

from dataclasses import dataclass

import xarray as xr

from aiweather.core import ForecastMetadata


@dataclass(slots=True)
class Forecast:
    """
    Container for an AIWeather forecast.
    """

    dataset: xr.Dataset
    metadata: ForecastMetadata

    def __repr__(self) -> str:        
        return (
            "Forecast("
            f"model={self.metadata.model_name!r}, "
            f"forecast_id={self.metadata.forecast_id!r}, "
            f"variables={list(self.dataset.data_vars)}, "
            f"sizes={dict(self.dataset.sizes)}"
            ")"
        )

    def to_xarray(self) -> xr.Dataset:
        """
        Return the underlying xarray dataset.
        """
        return self.dataset

    @property
    def variables(self) -> tuple[str, ...]:
        """
        Forecast variables.
        """
        return tuple(self.dataset.data_vars)

    @property
    def shape(self):
        """
        Dataset dimensions.
        """
        return self.dataset.sizes