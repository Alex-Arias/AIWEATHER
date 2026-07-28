"""
Forecast object for AIWeather.
"""

from __future__ import annotations

from dataclasses import dataclass

import xarray as xr

from .metadata import ForecastMetadata


@dataclass(slots=True)
class Forecast:
    """
    Container for an AIWeather forecast.

    Parameters
    ----------
    dataset : xr.Dataset
        Forecast stored as an xarray Dataset.

    metadata : ForecastMetadata
        Metadata describing the forecast.
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

    def __len__(self) -> int:
        """
        Number of forecast initialization times.
        """
        if "time" in self.dataset.dims:
            return self.dataset.sizes["time"]
        return 0

    def to_xarray(self) -> xr.Dataset:
        """
        Return the underlying xarray Dataset.
        """
        return self.dataset

    # ---------------------------------------------------------
    # Convenience properties
    # ---------------------------------------------------------

    @property
    def variables(self) -> tuple[str, ...]:
        """
        Forecast variables.
        """
        return tuple(self.dataset.data_vars)

    @property
    def coords(self):
        """
        Dataset coordinates.
        """
        return self.dataset.coords

    @property
    def dimensions(self):
        """
        Dataset dimensions.
        """
        return self.dataset.dims

    @property
    def shape(self):
        """
        Dataset sizes.
        """
        return self.dataset.sizes

    @property
    def latitude(self):
        """
        Latitude coordinate.
        """
        if "lat" in self.dataset.coords:
            return self.dataset["lat"]
        if "latitude" in self.dataset.coords:
            return self.dataset["latitude"]
        return None

    @property
    def longitude(self):
        """
        Longitude coordinate.
        """
        if "lon" in self.dataset.coords:
            return self.dataset["lon"]
        if "longitude" in self.dataset.coords:
            return self.dataset["longitude"]
        return None

    @property
    def lead_time(self):
        """
        Lead-time coordinate.
        """
        if "lead_time" in self.dataset.coords:
            return self.dataset["lead_time"]
        return None

    @property
    def initialization_time(self):
        """
        Forecast initialization time.
        """
        if "time" in self.dataset.coords:
            return self.dataset["time"]
        return None