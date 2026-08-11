"""
Forecast object and forecast loading utilities for AIWeather.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import re

import numpy as np
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
        """Number of forecast initialization times."""
        if "time" in self.dataset.dims:
            return self.dataset.sizes["time"]
        return 0

    def to_xarray(self) -> xr.Dataset:
        """Return the underlying xarray Dataset."""
        return self.dataset

    # ---------------------------------------------------------
    # Convenience properties
    # ---------------------------------------------------------

    @property
    def variables(self) -> tuple[str, ...]:
        """Forecast variables."""
        return tuple(self.dataset.data_vars)

    @property
    def coords(self):
        """Dataset coordinates."""
        return self.dataset.coords

    @property
    def dimensions(self):
        """Dataset dimensions."""
        return self.dataset.dims

    @property
    def shape(self):
        """Dataset sizes."""
        return self.dataset.sizes

    @property
    def latitude(self):
        """Latitude coordinate."""
        if "lat" in self.dataset.coords:
            return self.dataset["lat"]

        if "latitude" in self.dataset.coords:
            return self.dataset["latitude"]

        return None

    @property
    def longitude(self):
        """Longitude coordinate."""
        if "lon" in self.dataset.coords:
            return self.dataset["lon"]

        if "longitude" in self.dataset.coords:
            return self.dataset["longitude"]

        return None

    @property
    def lead_time(self):
        """Lead-time coordinate."""
        if "lead_time" in self.dataset.coords:
            return self.dataset["lead_time"]

        return None

    @property
    def initialization_time(self):
        """Forecast initialization time."""
        if "time" in self.dataset.coords:
            return self.dataset["time"]

        return None

    # ---------------------------------------------------------
    # Variable access
    # ---------------------------------------------------------

    def get_variable(
        self,
        name: str,
        *,
        lead_time: int | None = None,
    ) -> xr.DataArray:
        """
        Return a forecast variable.

        Parameters
        ----------
        name : str
            Name of the forecast variable.

        lead_time : int, optional
            Forecast lead time in hours. If omitted, the complete
            variable is returned.

        Returns
        -------
        xr.DataArray
            Requested forecast variable.

        Raises
        ------
        KeyError
            If the requested variable does not exist.

        ValueError
            If the requested lead time is not available.
        """

        if name not in self.dataset.data_vars:
            raise KeyError(
                f"Forecast variable {name!r} not found. "
                f"Available variables: {list(self.dataset.data_vars)}"
            )

        variable = self.dataset[name]

        if lead_time is None:
            return variable

        if "lead_time" not in variable.dims:
            raise ValueError(
                f"Variable {name!r} does not contain a lead_time dimension."
            )

        # The Zarr dataset stores lead_time as timedelta64[h].
        # Convert it explicitly to integer hours before comparison.
        lead_hours = (
            variable["lead_time"]
            .values
            .astype("timedelta64[h]")
            .astype(int)
        )

        matches = np.flatnonzero(lead_hours == lead_time)

        if len(matches) == 0:
            available = lead_hours.tolist()

            raise ValueError(
                f"Lead time {lead_time} h is not available. "
                f"Available lead times: {available}"
            )

        # Keep lead time as a singleton dimension
        return variable.isel(
            lead_time=[int(matches[0])]
        )


    # ---------------------------------------------------------
    # Lead-time selection
    # ---------------------------------------------------------

    def select_lead_time(self, lead_time: int) -> Forecast:
        """
        Return a new Forecast containing one selected lead time.

        Parameters
        ----------
        lead_time : int
            Forecast lead time in hours.

        Returns
        -------
        Forecast
            Forecast containing the selected lead time.

        Raises
        ------
        ValueError
            If the requested lead time is not available.
        """

        if "lead_time" not in self.dataset.coords:
            raise ValueError(
                "Forecast dataset does not contain a lead_time coordinate."
            )

        # Convert timedelta64[h] values to integer hours before
        # comparing with the user-supplied integer lead time.
        lead_hours = (
            self.dataset["lead_time"]
            .values
            .astype("timedelta64[h]")
            .astype(int)
        )

        matches = np.flatnonzero(lead_hours == lead_time)

        if len(matches) == 0:
            available = lead_hours.tolist()

            raise ValueError(
                f"Lead time {lead_time} h is not available. "
                f"Available lead times: {available}"
            )

        dataset = self.dataset.isel(
            lead_time=slice(
                int(matches[0]),
                int(matches[0]) + 1,
            )
        )

        return Forecast(
            dataset=dataset,
            metadata=self.metadata,
        )


def _metadata_from_path(
    path: str | Path,
    dataset: xr.Dataset,
) -> ForecastMetadata:
    """
    Build ForecastMetadata from the standardized AIWeather output path.
    """

    path = Path(path)
    name = path.name

    pattern = re.compile(
        r"^(?P<model>[^_]+)_(?P<datasource>[^_]+)_"
        r"(?P<init>\d{8}T\d{6})_(?P<hours>\d+)h\.zarr$"
    )

    match = pattern.match(name)

    if match is None:
        raise ValueError(
            "Forecast path does not match the expected AIWeather "
            "output format: "
            "<model>_<datasource>_<YYYYMMDDTHHMMSS>_<hours>h.zarr"
        )

    model = match.group("model")
    init_string = match.group("init")

    initialization_time = datetime.strptime(
        init_string,
        "%Y%m%dT%H%M%S",
    )

    forecast_id = path.stem

    return ForecastMetadata(
        model_name=model,
        model_version="unknown",
        backend="earth2studio",
        forecast_id=forecast_id,
        initialization_time=initialization_time,
        creation_time=None,
    )


def open_forecast(
    path: str | Path,
    *,
    consolidated: bool = True,
) -> Forecast:
    """
    Open an AIWeather forecast stored in Zarr format.

    Parameters
    ----------
    path : str or pathlib.Path
        Path to the forecast Zarr store.

    consolidated : bool, default=True
        Whether to read consolidated Zarr metadata.

    Returns
    -------
    Forecast
        AIWeather Forecast object containing a lazy xarray Dataset.
    """

    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"Forecast Zarr store not found: {path}"
        )

    dataset = xr.open_zarr(
        path,
        consolidated=consolidated,
    )

    metadata = _metadata_from_path(
        path,
        dataset,
    )

    return Forecast(
        dataset=dataset,
        metadata=metadata,
    )
