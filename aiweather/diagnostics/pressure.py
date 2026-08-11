"""
Pressure diagnostics for AIWeather.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import xarray as xr


@dataclass(frozen=True, slots=True)
class PressureMinimum:
    """
    Location and value of a pressure minimum.

    Attributes
    ----------
    value : float
        Minimum pressure value.

    latitude : float
        Latitude of the minimum in degrees.

    longitude : float
        Longitude of the minimum in the coordinate convention
        used by the input DataArray.

    units : str or None
        Pressure units copied from the input field when available.
    """

    value: float
    latitude: float
    longitude: float
    units: str | None = None


def pressure_minimum(
    pressure: xr.DataArray,
) -> PressureMinimum:
    """
    Find the minimum of a two-dimensional pressure field.

    Parameters
    ----------
    pressure : xr.DataArray
        Pressure field containing latitude and longitude dimensions.

        The field should represent one forecast time. Latitude and
        longitude coordinate names may be either ``lat`` / ``lon``
        or ``latitude`` / ``longitude``.

    Returns
    -------
    PressureMinimum
        Minimum pressure value and its grid-point coordinates.

    Raises
    ------
    TypeError
        If pressure is not an xarray DataArray.

    ValueError
        If latitude or longitude coordinates are missing, if
        non-spatial dimensions contain more than one element,
        or if the field contains no finite values.
    """
    if not isinstance(pressure, xr.DataArray):
        raise TypeError(
            "pressure must be an xarray.DataArray, "
            f"got {type(pressure).__name__}"
        )

    if "lat" in pressure.coords:
        latitude_name = "lat"
    elif "latitude" in pressure.coords:
        latitude_name = "latitude"
    else:
        raise ValueError(
            "Pressure field does not contain a latitude coordinate."
        )

    if "lon" in pressure.coords:
        longitude_name = "lon"
    elif "longitude" in pressure.coords:
        longitude_name = "longitude"
    else:
        raise ValueError(
            "Pressure field does not contain a longitude coordinate."
        )

    # Permit extra singleton dimensions such as time or lead_time,
    # but reject ambiguous multi-time fields.
    non_spatial_dims = [
        dim
        for dim in pressure.dims
        if dim not in (latitude_name, longitude_name)
    ]

    for dim in non_spatial_dims:
        if pressure.sizes[dim] != 1:
            raise ValueError(
                "pressure_minimum requires a single forecast time. "
                f"Dimension {dim!r} has size {pressure.sizes[dim]}."
            )

    field = pressure.squeeze(
        dim=non_spatial_dims,
        drop=True,
    )

    if latitude_name not in field.dims:
        raise ValueError(
            "Latitude coordinate must be a dimension of the "
            "pressure field."
        )

    if longitude_name not in field.dims:
        raise ValueError(
            "Longitude coordinate must be a dimension of the "
            "pressure field."
        )

    # Stack the horizontal grid so one argmin identifies both
    # latitude and longitude.
    stacked = field.stack(
        point=(latitude_name, longitude_name)
    )

    finite_count = (
        np.isfinite(stacked)
        .sum()
        .compute()
        .item()
    )

    if finite_count == 0:
        raise ValueError(
            "Pressure field contains no finite values."
        )

    minimum_index = int(
        stacked.argmin(
            dim="point",
            skipna=True,
        )
        .compute()
        .item()
    )

    minimum = stacked.isel(
        point=minimum_index
    ).compute()

    return PressureMinimum(
        value=float(minimum.item()),
        latitude=float(
            minimum[latitude_name].item()
        ),
        longitude=float(
            minimum[longitude_name].item()
        ),
        units=pressure.attrs.get("units"),
    )
