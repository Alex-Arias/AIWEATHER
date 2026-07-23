"""
ERA5 dataset implementation.

This module provides the ERA5 dataset interface used throughout
AIWeather. It implements the abstract Dataset API while remaining
independent of any specific backend (NetCDF, Zarr, etc.).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import xarray as xr

from aiweather.io import open_dataset

from .dataset import Dataset
from .metadata import DatasetMetadata


class ERA5Dataset(Dataset):
    """
    ERA5 meteorological dataset.
    """

    def __init__(self, path: str | Path):
        metadata = DatasetMetadata(
            name="ERA5",
            source="ECMWF",
            file_format="NetCDF",
            spatial_resolution="0.25°",
            temporal_resolution="1 hour",
        )

        super().__init__(path)

        self.metadata = metadata
        self._dataset: xr.Dataset | None = None

    def open(self) -> None:
        """
        Open the dataset.
        """
        if not self.path.exists():
            raise FileNotFoundError(
                f"Dataset not found: {self.path}"
            )

        self._dataset = open_dataset(self.path)

    def close(self) -> None:
        """
        Close the dataset.
        """
        if self._dataset is not None:
            self._dataset.close()

        self._dataset = None

    @property
    def dataset(self) -> xr.Dataset:
        """
        Return the loaded xarray dataset.

        Raises
        ------
        RuntimeError
            If the dataset has not been opened.
        """
        if self._dataset is None:
            raise RuntimeError(
                "Dataset has not been opened."
            )

        return self._dataset

    def variables(self) -> list[str]:
        """
        Return dataset variables.
        """
        return list(self.dataset.data_vars)

    def times(self):
        """
        Return available timestamps.
        """
        if "time" in self.dataset.coords:
            return self.dataset["time"]

        return None

    def grid(self):
        """
        Return latitude and longitude coordinates.
        """
        grid = {}

        if "latitude" in self.dataset.coords:
            grid["latitude"] = self.dataset["latitude"]

        if "longitude" in self.dataset.coords:
            grid["longitude"] = self.dataset["longitude"]

        return grid

    def subset(self, **kwargs: Any) -> Any:
        """
        Return a subset of the dataset.
        """
        raise NotImplementedError(
            "Subset functionality will be implemented "
            "after xarray integration."
        )