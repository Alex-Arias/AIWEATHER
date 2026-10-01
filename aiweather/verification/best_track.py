"""
Best-track observation utilities.

Provides standardized tropical cyclone best-track observations and
conversion to AIWeather TrackRecord objects for forecast verification.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from aiweather.tracking.records import TrackRecord


@dataclass(slots=True)
class BestTrackPoint:
    """
    One tropical cyclone best-track observation.

    Parameters
    ----------
    valid_time
        Observation valid time.

    latitude
        Storm-center latitude in degrees north.

    longitude
        Storm-center longitude in degrees east.

    pressure
        Minimum sea-level pressure.

    max_wind
        Maximum sustained wind.

    pressure_units
        Units associated with pressure.

    wind_units
        Units associated with maximum wind.
    """

    valid_time: np.datetime64
    latitude: float
    longitude: float
    pressure: float | None = None
    max_wind: float | None = None
    pressure_units: str | None = None
    wind_units: str | None = None


def best_track_to_records(
    points: list[BestTrackPoint],
    *,
    initialization_time,
) -> list[TrackRecord]:
    """
    Convert best-track observations to AIWeather TrackRecord objects.

    Forecast lead time is calculated relative to the supplied forecast
    initialization time.

    Parameters
    ----------
    points
        Best-track observations.

    initialization_time
        Forecast initialization time.

    Returns
    -------
    list[TrackRecord]
        Best-track observations represented using the standard
        AIWeather tracking record structure.
    """

    if not isinstance(
        points,
        list,
    ):
        raise TypeError(
            "points must be a list."
        )

    for point in points:
        if not isinstance(
            point,
            BestTrackPoint,
        ):
            raise TypeError(
                "points must contain BestTrackPoint objects."
            )

    if (
        isinstance(initialization_time, str)
        and len(initialization_time) == 15
        and initialization_time[8] == "T"
    ):
        initialization_time = (
            f"{initialization_time[:4]}-"
            f"{initialization_time[4:6]}-"
            f"{initialization_time[6:8]}T"
            f"{initialization_time[9:11]}:"
            f"{initialization_time[11:13]}:"
            f"{initialization_time[13:15]}"
        )

    initialization = np.datetime64(
        initialization_time
    )

    if np.isnat(initialization):
        raise ValueError(
            "initialization_time cannot be NaT."
        )

    records = []

    for point in sorted(
        points,
        key=lambda item: item.valid_time,
    ):
        valid_time = np.datetime64(
            point.valid_time
        )

        if np.isnat(valid_time):
            raise ValueError(
                "Best-track valid_time cannot be NaT."
            )

        delta = (
            valid_time
            - initialization
        )

        lead_time_hours = int(
            delta
            / np.timedelta64(
                1,
                "h",
            )
        )

        records.append(
            TrackRecord(
                lead_time_hours=lead_time_hours,
                valid_time=valid_time,
                latitude=float(
                    point.latitude
                ),
                longitude=float(
                    point.longitude
                ),
                pressure=(
                    None
                    if point.pressure is None
                    else float(
                        point.pressure
                    )
                ),
                pressure_units=(
                    point.pressure_units
                ),
                max_wind=(
                    None
                    if point.max_wind is None
                    else float(
                        point.max_wind
                    )
                ),
                wind_units=(
                    point.wind_units
                ),
            )
        )

    return records