"""
Automatic tropical-cyclone genesis detection for AIWeather.

This module evaluates associated candidate tracks using configurable
pressure, wind, and persistence criteria.

The detected genesis point is the first point belonging to a
persistent qualifying sequence. This is a research-oriented
candidate-genesis diagnostic, not an operational classification.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import xarray as xr

from aiweather.diagnostics import wind_speed

from .association import CandidateTrack
from .tropical_cyclone import great_circle_distance_km


@dataclass(frozen=True, slots=True)
class GenesisResult:
    """
    Genesis result for one associated candidate track.

    Attributes
    ----------
    track_index : int
        Index of the associated CandidateTrack.

    genesis_lead_time_hours : int
        Forecast lead time of the first persistent qualifying point.

    latitude : float
        Genesis latitude.

    longitude : float
        Genesis longitude.

    pressure : float
        Center pressure at genesis.

    max_wind : float
        Maximum local wind speed around the genesis center.

    qualifying_points : int
        Number of points belonging to the persistent qualifying run.
    """

    track_index: int
    genesis_lead_time_hours: int
    latitude: float
    longitude: float
    pressure: float
    max_wind: float
    qualifying_points: int


def _validate_wind_fields(
    u_wind: xr.DataArray,
    v_wind: xr.DataArray,
) -> None:
    """
    Validate wind fields used for genesis detection.
    """
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
            "u_wind and v_wind must have identical "
            "coordinates and dimensions."
        ) from exc

    if "lead_time" not in u_wind.dims:
        raise ValueError(
            "Wind fields must contain a 'lead_time' dimension."
        )

    if "time" in u_wind.dims:
        if u_wind.sizes["time"] != 1:
            raise ValueError(
                "Genesis detection currently requires exactly "
                "one forecast initialization time."
            )


def _lead_time_hours(
    lead_time: xr.DataArray,
) -> np.ndarray:
    """
    Convert lead-time coordinate values to integer hours.
    """
    return (
        lead_time.values
        .astype("timedelta64[h]")
        .astype(int)
    )


def _local_max_wind(
    u: xr.DataArray,
    v: xr.DataArray,
    *,
    latitude_grid: xr.DataArray,
    longitude_grid: xr.DataArray,
    center_latitude: float,
    center_longitude: float,
    radius_km: float,
) -> float:
    """
    Calculate maximum wind speed near a candidate center.
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

    value = float(
        maximum.item()
    )

    if not np.isfinite(value):
        raise ValueError(
            "No finite wind values were found within the "
            "requested wind radius."
        )

    return value


def _persistent_runs(
    flags: list[bool],
    minimum_consecutive_points: int,
) -> list[tuple[int, int]]:
    """
    Return inclusive-exclusive persistent runs of True values.
    """
    runs: list[tuple[int, int]] = []

    start = None

    for index, value in enumerate(
        flags + [False]
    ):
        if value and start is None:
            start = index

        elif not value and start is not None:
            if (
                index - start
                >= minimum_consecutive_points
            ):
                runs.append(
                    (
                        start,
                        index,
                    )
                )

            start = None

    return runs


def detect_genesis(
    candidate_tracks: list[CandidateTrack],
    *,
    u_wind: xr.DataArray,
    v_wind: xr.DataArray,
    minimum_wind: float,
    maximum_pressure: float,
    minimum_consecutive_points: int = 3,
    wind_radius_km: float = 300.0,
) -> list[GenesisResult]:
    """
    Detect persistent genesis candidates from associated tracks.

    Parameters
    ----------
    candidate_tracks : list[CandidateTrack]
        Associated pressure-minimum trajectories.

    u_wind, v_wind : xr.DataArray
        Zonal and meridional wind fields containing lead_time,
        latitude, and longitude coordinates.

    minimum_wind : float
        Minimum local maximum wind required.

    maximum_pressure : float
        Maximum allowed center pressure.

    minimum_consecutive_points : int, default=3
        Minimum number of consecutive qualifying points required
        to define a persistent genesis candidate.

    wind_radius_km : float, default=300
        Radius used to calculate local maximum wind around each
        candidate center.

    Returns
    -------
    list[GenesisResult]
        One genesis result for every associated track that contains
        a persistent qualifying sequence.

    Raises
    ------
    TypeError
        If candidate_tracks or wind fields have invalid types.

    ValueError
        If thresholds, dimensions, or coordinates are invalid.
    """
    if not isinstance(candidate_tracks, list):
        raise TypeError(
            "candidate_tracks must be a list of CandidateTrack "
            "objects."
        )

    for track in candidate_tracks:
        if not isinstance(track, CandidateTrack):
            raise TypeError(
                "candidate_tracks must contain only "
                "CandidateTrack objects."
            )

    if minimum_wind < 0.0:
        raise ValueError(
            "minimum_wind must be greater than or equal to zero."
        )

    if not np.isfinite(maximum_pressure):
        raise ValueError(
            "maximum_pressure must be finite."
        )

    if minimum_consecutive_points < 1:
        raise ValueError(
            "minimum_consecutive_points must be at least 1."
        )

    if wind_radius_km <= 0.0:
        raise ValueError(
            "wind_radius_km must be greater than zero."
        )

    _validate_wind_fields(
        u_wind,
        v_wind,
    )

    if not candidate_tracks:
        return []

    if "lat" in u_wind.coords:
        latitude_name = "lat"
    elif "latitude" in u_wind.coords:
        latitude_name = "latitude"
    else:
        raise ValueError(
            "Wind fields do not contain a latitude coordinate."
        )

    if "lon" in u_wind.coords:
        longitude_name = "lon"
    elif "longitude" in u_wind.coords:
        longitude_name = "longitude"
    else:
        raise ValueError(
            "Wind fields do not contain a longitude coordinate."
        )

    if "time" in u_wind.dims:
        u_wind = u_wind.isel(
            time=0
        )

        v_wind = v_wind.isel(
            time=0
        )

    available_leads = _lead_time_hours(
        u_wind["lead_time"]
    )

    lead_index = {
        int(value): index
        for index, value in enumerate(
            available_leads
        )
    }

    latitude = u_wind[
        latitude_name
    ]

    longitude = u_wind[
        longitude_name
    ]

    latitude_grid, longitude_grid = xr.broadcast(
        latitude,
        longitude,
    )

    results: list[GenesisResult] = []

    for track_index, track in enumerate(
        candidate_tracks
    ):
        if len(track) == 0:
            continue

        wind_values: list[float] = []
        qualifying: list[bool] = []

        for point in track.points:
            if point.lead_time_hours not in lead_index:
                raise ValueError(
                    f"Lead time {point.lead_time_hours} h is "
                    "not available in the wind fields."
                )

            index = lead_index[
                point.lead_time_hours
            ]

            u = u_wind.isel(
                lead_time=index
            )

            v = v_wind.isel(
                lead_time=index
            )

            maximum_wind = _local_max_wind(
                u,
                v,
                latitude_grid=latitude_grid,
                longitude_grid=longitude_grid,
                center_latitude=point.latitude,
                center_longitude=point.longitude,
                radius_km=wind_radius_km,
            )

            wind_values.append(
                maximum_wind
            )

            qualifying.append(
                bool(
                    point.pressure
                    <= maximum_pressure
                    and maximum_wind
                    >= minimum_wind
                )
            )

        runs = _persistent_runs(
            qualifying,
            minimum_consecutive_points,
        )

        if not runs:
            continue

        start, end = runs[0]

        genesis_point = track.points[
            start
        ]

        results.append(
            GenesisResult(
                track_index=track_index,
                genesis_lead_time_hours=(
                    genesis_point.lead_time_hours
                ),
                latitude=genesis_point.latitude,
                longitude=genesis_point.longitude,
                pressure=genesis_point.pressure,
                max_wind=wind_values[start],
                qualifying_points=end - start,
            )
        )

    return results


def select_first_genesis(
    results: list[GenesisResult],
) -> GenesisResult | None:
    """
    Select the earliest detected genesis result.

    If multiple tracks have genesis at the same lead time, the
    result with the lower center pressure is selected.
    """
    if not isinstance(results, list):
        raise TypeError(
            "results must be a list of GenesisResult objects."
        )

    for result in results:
        if not isinstance(result, GenesisResult):
            raise TypeError(
                "results must contain only GenesisResult objects."
            )

    if not results:
        return None

    return min(
        results,
        key=lambda result: (
            result.genesis_lead_time_hours,
            result.pressure,
        ),
    )
