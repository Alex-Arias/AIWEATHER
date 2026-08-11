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

def _great_circle_distance_scalar_km(
    latitude1: float,
    longitude1: float,
    latitude2: float,
    longitude2: float,
) -> float:
    """
    Calculate great-circle distance between two scalar points.
    """
    earth_radius_km = 6371.0

    lat1 = np.deg2rad(latitude1)
    lat2 = np.deg2rad(latitude2)

    lon1 = np.deg2rad(longitude1)
    lon2 = np.deg2rad(longitude2)

    dlat = lat2 - lat1

    dlon = (
        lon2
        - lon1
        + np.pi
    ) % (2.0 * np.pi) - np.pi

    a = (
        np.sin(dlat / 2.0) ** 2
        + np.cos(lat1)
        * np.cos(lat2)
        * np.sin(dlon / 2.0) ** 2
    )

    return float(
        2.0
        * earth_radius_km
        * np.arcsin(np.sqrt(a))
    )


def detect_pressure_minima(
    pressure: xr.DataArray,
    *,
    max_candidates: int = 10,
    minimum_separation_km: float = 500.0,
) -> list[PressureMinimum]:
    """
    Detect multiple spatially separated local pressure minima.

    Local minima are identified on the horizontal grid using the
    eight surrounding grid cells. Candidates are ranked from lowest
    to highest pressure, then filtered so accepted minima are at least
    ``minimum_separation_km`` apart.

    Parameters
    ----------
    pressure : xr.DataArray
        Pressure field containing latitude and longitude dimensions.
        The field should represent one forecast time. Additional
        singleton dimensions such as ``time`` or ``lead_time`` are
        permitted.

    max_candidates : int, default=10
        Maximum number of minima to return.

    minimum_separation_km : float, default=500
        Minimum allowed great-circle distance between returned
        candidate minima.

    Returns
    -------
    list[PressureMinimum]
        Spatially separated pressure minima ordered from lowest
        pressure to highest pressure.

    Raises
    ------
    TypeError
        If pressure is not an xarray DataArray.

    ValueError
        If coordinates are missing, non-spatial dimensions contain
        multiple values, parameters are invalid, or no finite values
        are available.
    """
    if not isinstance(pressure, xr.DataArray):
        raise TypeError(
            "pressure must be an xarray.DataArray, "
            f"got {type(pressure).__name__}"
        )

    if not isinstance(max_candidates, int):
        raise TypeError(
            "max_candidates must be an integer."
        )

    if max_candidates < 1:
        raise ValueError(
            "max_candidates must be at least 1."
        )

    if minimum_separation_km < 0.0:
        raise ValueError(
            "minimum_separation_km must be greater than "
            "or equal to zero."
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

    non_spatial_dims = [
        dim
        for dim in pressure.dims
        if dim not in (
            latitude_name,
            longitude_name,
        )
    ]

    for dim in non_spatial_dims:
        if pressure.sizes[dim] != 1:
            raise ValueError(
                "detect_pressure_minima requires a single "
                "forecast time. "
                f"Dimension {dim!r} has size "
                f"{pressure.sizes[dim]}."
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

    # Put the horizontal dimensions in a predictable order.
    field = field.transpose(
        latitude_name,
        longitude_name,
    )

    values = np.asarray(
        field.compute().values,
        dtype=float,
    )

    finite = np.isfinite(values)

    if not finite.any():
        raise ValueError(
            "Pressure field contains no finite values."
        )

    # Pad with +infinity so valid minima located along the edge
    # can still be detected.
    padded = np.pad(
        values,
        pad_width=1,
        mode="constant",
        constant_values=np.inf,
    )

    center = padded[1:-1, 1:-1]

    neighbors = [
        padded[:-2, :-2],
        padded[:-2, 1:-1],
        padded[:-2, 2:],
        padded[1:-1, :-2],
        padded[1:-1, 2:],
        padded[2:, :-2],
        padded[2:, 1:-1],
        padded[2:, 2:],
    ]

    local_minimum_mask = finite.copy()

    for neighbor in neighbors:
        local_minimum_mask &= (
            center <= neighbor
        )

    candidate_indices = np.argwhere(
        local_minimum_mask
    )

    if candidate_indices.size == 0:
        return []

    # Rank candidates from lowest pressure to highest pressure.
    ranked = sorted(
        candidate_indices,
        key=lambda index: values[
            int(index[0]),
            int(index[1]),
        ],
    )

    latitude_values = field[
        latitude_name
    ].values

    longitude_values = field[
        longitude_name
    ].values

    units = pressure.attrs.get(
        "units"
    )

    accepted: list[PressureMinimum] = []

    for latitude_index, longitude_index in ranked:
        latitude_index = int(
            latitude_index
        )

        longitude_index = int(
            longitude_index
        )

        candidate = PressureMinimum(
            value=float(
                values[
                    latitude_index,
                    longitude_index,
                ]
            ),
            latitude=float(
                latitude_values[
                    latitude_index
                ]
            ),
            longitude=float(
                longitude_values[
                    longitude_index
                ]
            ),
            units=units,
        )

        separated = True

        for existing in accepted:
            distance = (
                _great_circle_distance_scalar_km(
                    candidate.latitude,
                    candidate.longitude,
                    existing.latitude,
                    existing.longitude,
                )
            )

            if distance < minimum_separation_km:
                separated = False
                break

        if not separated:
            continue

        accepted.append(
            candidate
        )

        if len(accepted) >= max_candidates:
            break

    return accepted
