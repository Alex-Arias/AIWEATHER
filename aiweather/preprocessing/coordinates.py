"""
Coordinate utilities for AIWeather preprocessing.
"""

from __future__ import annotations

import xarray as xr


def sort_latitude(dataset: xr.Dataset) -> xr.Dataset:
    """
    Ensure latitude is ordered from north to south.

    GraphCast expects:

        90
        ...
        -90
    """

    if dataset.lat[0] < dataset.lat[-1]:
        dataset = dataset.sortby("lat", ascending=False)

    return dataset

def convert_longitude(dataset: xr.Dataset) -> xr.Dataset:
    """
    Convert longitudes from [-180, 180] to [0, 360].

    GraphCast expects:

        0
        ...
        359.75
    """

    lon = dataset.lon

    # Already in 0–360 convention
    if lon.min() >= 0:
        return dataset

    dataset = dataset.assign_coords(
        lon=((lon + 360) % 360)
    )

    dataset = dataset.sortby("lon")

    return dataset    

def ensure_coordinate_order(dataset: xr.Dataset) -> xr.Dataset:
    """
    Ensure dataset dimensions follow the standard order.

    Preferred order:

        time
        lat
        lon

    Dimensions that do not exist are ignored.
    """

    preferred = [
        dim
        for dim in ("time", "lat", "lon")
        if dim in dataset.dims
    ]

    remaining = [
        dim
        for dim in dataset.dims
        if dim not in preferred
    ]

    return dataset.transpose(
        *(preferred + remaining)
    )