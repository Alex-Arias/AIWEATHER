"""
Dataset backend utilities.

This module provides a unified interface for opening meteorological
datasets independently of the underlying storage format.
"""

from __future__ import annotations

from pathlib import Path

import xarray as xr


_ENGINE_MAP = {
    ".nc": "netcdf4",
    ".nc4": "netcdf4",
    ".cdf": "netcdf4",
    ".zarr": "zarr",
    ".grib": "cfgrib",
    ".grib2": "cfgrib",
    ".grb": "cfgrib",
    ".grb2": "cfgrib",
}


def detect_engine(path: str | Path) -> str:
    """
    Determine the appropriate xarray backend.

    Parameters
    ----------
    path
        Dataset path.

    Returns
    -------
    str
        Xarray engine name.
    """

    suffix = Path(path).suffix.lower()

    if suffix not in _ENGINE_MAP:
        raise ValueError(
            f"Unsupported dataset format: '{suffix}'"
        )

    return _ENGINE_MAP[suffix]


def open_dataset(path: str | Path) -> xr.Dataset:
    """
    Open a dataset using the appropriate backend.

    Parameters
    ----------
    path
        Dataset path.

    Returns
    -------
    xarray.Dataset
    """

    path = Path(path)

    engine = detect_engine(path)

    return xr.open_dataset(path, engine=engine)