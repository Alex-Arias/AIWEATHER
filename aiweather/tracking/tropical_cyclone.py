"""
Tropical-cyclone tracking primitives for AIWeather.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import xarray as xr

from aiweather.diagnostics import (
    pressure_minimum,
    wind_speed,
)


EARTH_RADIUS_KM = 6371.0


@dataclass(frozen=True, slots=True)
class TrackPoint:
    """
    One candidate cyclone-center position.

    Attributes
    ----------
    lead_time_hours : int
        Forecast lead time in hours.

    latitude : float
        Candidate center latitude in degrees.

    longitude : float
        Candidate center longitude in the forecast coordinate
        convention.

    pressure : float
        Minimum sea-level pressure at the candidate center.

    pressure_units : str or None
        Pressure units copied from the source field when available.

    max_wind : float or None
        Maximum horizontal wind speed within the configured wind
        search radius.

    wind_units : str or None
        Wind-speed units when available from the source components.
    """

    lead_time_hours: int
    latitude: float
    longitude: float
    pressure: float
    pressure_units: str | None = None
    max_wind: float | None = None
    wind_units: str | None = None


def great_circle_distance_km(
    latitude: xr.DataArray | np.ndarray | float,
    longitude: xr.DataArray | np.ndarray | float,
    center_latitude: float,
    center_longitude: float,
):
    """
    Calculate great-circle distance from a reference point.

    Parameters
    ----------
    latitude, longitude
        Coordinates of points to evaluate.

    center_latitude, center_longitude : float
        Reference point in degrees.

    Returns
    -------
    array-like
        Great-circle distance in kilometers.
    """
    lat1 = np.deg2rad(latitude)
    lon1 = np.deg2rad(longitude)

    lat2 = np.deg2rad(center_latitude)
    lon2 = np.deg2rad(center_longitude)

    dlat = lat1 - lat2

    # Wrap longitude difference to [-pi, pi].
    dlon = (
        lon1
        - lon2
        + np.pi
    ) % (2.0 * np.pi) - np.pi

    a = (
        np.sin(dlat / 2.0) ** 2
        + np.cos(lat2)
        * np.cos(lat1)
        * np.sin(dlon / 2.0) ** 2
    )

    return (
        2.0
        * EARTH_RADIUS_KM
        * np.arcsin(np.sqrt(a))
    )


def _validate_wind_fields(
    u_wind: xr.DataArray | None,
    v_wind: xr.DataArray | None,
) -> None:
    """
    Validate optional wind fields supplied to the tracker.
    """
    if (u_wind is None) != (v_wind is None):
        raise ValueError(
            "u_wind and v_wind must be provided together."
        )

    if u_wind is None:
        return

    if not isinstance(u_wind, xr.DataArray):
        raise TypeError(
            "u_wind must be an xarray.DataArray."
        )

    if not isinstance(v_wind, xr.DataArray):
        raise TypeError(
            "v_wind must be an xarray.DataArray."
        )

    try:
        xr.align(
            u_wind,
            v_wind,
            join="exact",
        )
    except ValueError as exc:
        raise ValueError(
            "u_wind and v_wind must have identical coordinates "
            "and dimensions."
        ) from exc


def _local_max_wind(
    u: xr.DataArray,
    v: xr.DataArray,
    *,
    latitude_grid: xr.DataArray,
    longitude_grid: xr.DataArray,
    center_latitude: float,
    center_longitude: float,
    radius_km: float,
) -> tuple[float, str | None]:
    """
    Calculate maximum wind speed within a radius of a center.
    """
    speed = wind_speed(
        u,
        v,
    )

    distance = great_circle_distance_km(
        latitude_grid,
        longitude_grid,
        center_latitude,
        center_longitude,
    )

    local_speed = speed.where(
        distance <= radius_km
    )

    maximum = local_speed.max(
        skipna=True
    ).compute()

    if not np.isfinite(maximum.item()):
        raise ValueError(
            "No finite wind values were found inside the "
            "requested wind search radius."
        )

    return (
        float(maximum.item()),
        speed.attrs.get("units"),
    )


def track_pressure_minimum(
    pressure: xr.DataArray,
    *,
    initial_latitude: float,
    initial_longitude: float,
    search_radius_km: float = 500.0,
    start_index: int = 0,
    u_wind: xr.DataArray | None = None,
    v_wind: xr.DataArray | None = None,
    wind_radius_km: float = 300.0,
    maximum_translation_speed_mps: float | None = None,
) -> list[TrackPoint]:
    """
    Track a pressure minimum using spatial continuity.

    At each forecast lead time, the pressure minimum is searched
    only within ``search_radius_km`` of the previous center.

    Optionally, zonal and meridional wind fields may be supplied.
    When present, the maximum horizontal wind speed within
    ``wind_radius_km`` of each candidate center is stored in the
    resulting TrackPoint.

    This routine produces candidate low-pressure-center tracks.
    It does not by itself determine whether the tracked system is
    a tropical cyclone.

    Parameters
    ----------
    pressure : xr.DataArray
        Sea-level pressure field with dimensions including
        lead_time, latitude, and longitude. A singleton time
        dimension is permitted.

    initial_latitude : float
        Seed-center latitude.

    initial_longitude : float
        Seed-center longitude in the same convention as the
        pressure field.

    search_radius_km : float, default=500
        Maximum displacement allowed between consecutive candidate
        centers.

    start_index : int, default=0
        Index of the first forecast lead time to track.

    u_wind, v_wind : xr.DataArray, optional
        Zonal and meridional wind components. Both must be supplied
        together.

    wind_radius_km : float, default=300
        Radius used to calculate local maximum wind speed.

    maximum_translation_speed_mps : float or None, default=None
        Optional maximum translation speed allowed between
        consecutive tracked centers. If exceeded, tracking
        terminates before the candidate center is accepted.
        None disables translation-speed termination.

    Returns
    -------
    list[TrackPoint]
        Candidate center track.

    Raises
    ------
    TypeError
        If required inputs are not xarray DataArrays.

    ValueError
        If required dimensions are missing or arguments are invalid.
    """
    if not isinstance(pressure, xr.DataArray):
        raise TypeError(
            "pressure must be an xarray.DataArray, "
            f"got {type(pressure).__name__}"
        )

    if search_radius_km <= 0.0:
        raise ValueError(
            "search_radius_km must be greater than zero."
        )

    if wind_radius_km <= 0.0:
        raise ValueError(
            "wind_radius_km must be greater than zero."
        )

    if (
        maximum_translation_speed_mps is not None
        and maximum_translation_speed_mps <= 0.0
    ):
        raise ValueError(
            "maximum_translation_speed_mps must be "
            "greater than zero."
        )

    if "lead_time" not in pressure.dims:
        raise ValueError(
            "pressure must contain a 'lead_time' dimension."
        )

    if "lat" in pressure.coords:
        latitude_name = "lat"
    elif "latitude" in pressure.coords:
        latitude_name = "latitude"
    else:
        raise ValueError(
            "pressure does not contain a latitude coordinate."
        )

    if "lon" in pressure.coords:
        longitude_name = "lon"
    elif "longitude" in pressure.coords:
        longitude_name = "longitude"
    else:
        raise ValueError(
            "pressure does not contain a longitude coordinate."
        )

    if (
        start_index < 0
        or start_index >= pressure.sizes["lead_time"]
    ):
        raise ValueError(
            f"start_index {start_index} is outside the available "
            "lead-time range."
        )

    _validate_wind_fields(
        u_wind,
        v_wind,
    )

    # A singleton initialization-time dimension is acceptable.
    if "time" in pressure.dims:
        if pressure.sizes["time"] != 1:
            raise ValueError(
                "track_pressure_minimum currently requires exactly "
                "one forecast initialization time."
            )

        pressure = pressure.isel(
            time=0
        )

    if u_wind is not None and "time" in u_wind.dims:
        if u_wind.sizes["time"] != 1:
            raise ValueError(
                "u_wind must contain exactly one initialization time."
            )

        u_wind = u_wind.isel(
            time=0
        )

        v_wind = v_wind.isel(
            time=0
        )

    if u_wind is not None:
        if "lead_time" not in u_wind.dims:
            raise ValueError(
                "Wind fields must contain a 'lead_time' dimension."
            )

        if (
            u_wind.sizes["lead_time"]
            != pressure.sizes["lead_time"]
        ):
            raise ValueError(
                "Pressure and wind fields must contain the same "
                "number of lead times."
            )

    lat = pressure[latitude_name]
    lon = pressure[longitude_name]

    lat2d, lon2d = xr.broadcast(
        lat,
        lon,
    )

    previous_latitude = float(
        initial_latitude
    )

    previous_longitude = float(
        initial_longitude
    )

    track: list[TrackPoint] = []

    for index in range(
        start_index,
        pressure.sizes["lead_time"],
    ):
        field = pressure.isel(
            lead_time=index
        )

        distance = great_circle_distance_km(
            lat2d,
            lon2d,
            previous_latitude,
            previous_longitude,
        )

        candidate = field.where(
            distance <= search_radius_km
        )

        if not bool(
            np.isfinite(candidate)
            .any()
            .compute()
            .item()
        ):
            break

        minimum = pressure_minimum(
            candidate
        )

        lead_hours = int(
            pressure["lead_time"]
            .values[index]
            .astype("timedelta64[h]")
            .astype(int)
        )

        if (
            maximum_translation_speed_mps is not None
            and track
        ):
            previous_point = track[-1]

            delta_hours = (
                lead_hours
                - previous_point.lead_time_hours
            )

            if delta_hours <= 0:
                raise ValueError(
                    "lead_time values must increase "
                    "strictly during tracking."
                )

            displacement_km = float(
                great_circle_distance_km(
                    previous_point.latitude,
                    previous_point.longitude,
                    minimum.latitude,
                    minimum.longitude,
                )
            )

            translation_speed_mps = (
                displacement_km * 1000.0
                / (delta_hours * 3600.0)
            )

            if (
                translation_speed_mps
                > maximum_translation_speed_mps
            ):
                break

        max_wind = None
        wind_units = None

        if u_wind is not None:
            u_field = u_wind.isel(
                lead_time=index
            )

            v_field = v_wind.isel(
                lead_time=index
            )

            max_wind, wind_units = _local_max_wind(
                u_field,
                v_field,
                latitude_grid=lat2d,
                longitude_grid=lon2d,
                center_latitude=minimum.latitude,
                center_longitude=minimum.longitude,
                radius_km=wind_radius_km,
            )

        point = TrackPoint(
            lead_time_hours=lead_hours,
            latitude=minimum.latitude,
            longitude=minimum.longitude,
            pressure=minimum.value,
            pressure_units=minimum.units,
            max_wind=max_wind,
            wind_units=wind_units,
        )

        track.append(
            point
        )

        previous_latitude = (
            point.latitude
        )

        previous_longitude = (
            point.longitude
        )

    return track

def track_from_genesis(
    pressure: xr.DataArray,
    *,
    genesis,
    u_wind: xr.DataArray | None = None,
    v_wind: xr.DataArray | None = None,
    search_radius_km: float = 500.0,
    wind_radius_km: float = 300.0,
) -> list[TrackPoint]:
    """
    Track a pressure minimum beginning from a detected genesis point.

    Parameters
    ----------
    pressure : xr.DataArray
        Sea-level pressure field containing a lead_time dimension.

    genesis
        GenesisResult-like object containing:
        ``genesis_lead_time_hours``, ``latitude``, and ``longitude``.

    u_wind, v_wind : xr.DataArray, optional
        Zonal and meridional wind components. Both must be supplied
        together when wind intensity should be attached to TrackPoint.

    search_radius_km : float, default=500
        Maximum displacement allowed between consecutive centers.

    wind_radius_km : float, default=300
        Radius used to calculate local maximum wind speed.

    Returns
    -------
    list[TrackPoint]
        Track beginning at the detected genesis lead time.

    Raises
    ------
    TypeError
        If genesis does not provide the required attributes.

    ValueError
        If the genesis lead time is not available in the pressure
        field.
    """
    required_attributes = (
        "genesis_lead_time_hours",
        "latitude",
        "longitude",
    )

    for attribute in required_attributes:
        if not hasattr(genesis, attribute):
            raise TypeError(
                "genesis must provide "
                "'genesis_lead_time_hours', "
                "'latitude', and 'longitude'."
            )

    if "lead_time" not in pressure.coords:
        raise ValueError(
            "pressure does not contain a 'lead_time' coordinate."
        )

    lead_hours = (
        pressure["lead_time"]
        .values
        .astype("timedelta64[h]")
        .astype(int)
    )

    matches = np.flatnonzero(
        lead_hours
        == int(genesis.genesis_lead_time_hours)
    )

    if len(matches) == 0:
        raise ValueError(
            "Genesis lead time "
            f"{genesis.genesis_lead_time_hours} h "
            "is not available in the pressure field."
        )

    start_index = int(matches[0])

    return track_pressure_minimum(
        pressure,
        initial_latitude=float(genesis.latitude),
        initial_longitude=float(genesis.longitude),
        search_radius_km=search_radius_km,
        start_index=start_index,
        u_wind=u_wind,
        v_wind=v_wind,
        wind_radius_km=wind_radius_km,
    )