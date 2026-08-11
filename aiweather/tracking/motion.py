"""
Tropical-cyclone track-motion diagnostics for AIWeather.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .tropical_cyclone import (
    TrackPoint,
    great_circle_distance_km,
)


@dataclass(frozen=True, slots=True)
class TrackMotion:
    """
    Motion diagnostics between consecutive track points.

    Attributes
    ----------
    lead_time_hours : int
        Forecast lead time of the destination track point.

    distance_km : float
        Great-circle distance traveled since the previous point.

    translation_speed_kmh : float
        Translation speed between consecutive points in km h-1.

    bearing_degrees : float
        Initial great-circle bearing measured clockwise from north.

    cumulative_distance_km : float
        Total distance traveled from the beginning of the track.
    """

    lead_time_hours: int
    distance_km: float
    translation_speed_kmh: float
    bearing_degrees: float
    cumulative_distance_km: float


def initial_bearing_degrees(
    latitude1: float,
    longitude1: float,
    latitude2: float,
    longitude2: float,
) -> float:
    """
    Calculate the initial great-circle bearing between two points.

    Parameters
    ----------
    latitude1, longitude1 : float
        Starting coordinates in degrees.

    latitude2, longitude2 : float
        Destination coordinates in degrees.

    Returns
    -------
    float
        Bearing in degrees clockwise from north in the range
        [0, 360).
    """
    lat1 = np.deg2rad(latitude1)
    lat2 = np.deg2rad(latitude2)

    lon1 = np.deg2rad(longitude1)
    lon2 = np.deg2rad(longitude2)

    # Wrap longitude difference to [-pi, pi].
    dlon = (
        lon2
        - lon1
        + np.pi
    ) % (2.0 * np.pi) - np.pi

    x = np.sin(dlon) * np.cos(lat2)

    y = (
        np.cos(lat1) * np.sin(lat2)
        - np.sin(lat1)
        * np.cos(lat2)
        * np.cos(dlon)
    )

    bearing = np.rad2deg(
        np.arctan2(x, y)
    )

    return float(
        bearing % 360.0
    )


def track_motion(
    track: list[TrackPoint],
) -> list[TrackMotion]:
    """
    Calculate motion diagnostics for a cyclone track.

    Motion is calculated between consecutive TrackPoint objects.
    Therefore a track containing N points produces N - 1 motion
    records.

    Parameters
    ----------
    track : list[TrackPoint]
        Candidate cyclone-center track.

    Returns
    -------
    list[TrackMotion]
        Motion diagnostics for consecutive track segments.

    Raises
    ------
    TypeError
        If track is not a list or contains objects other than
        TrackPoint.

    ValueError
        If consecutive lead times are not strictly increasing.
    """
    if not isinstance(track, list):
        raise TypeError(
            "track must be a list of TrackPoint objects."
        )

    for point in track:
        if not isinstance(point, TrackPoint):
            raise TypeError(
                "track must contain only TrackPoint objects."
            )

    if len(track) < 2:
        return []

    result: list[TrackMotion] = []

    cumulative_distance = 0.0

    for previous, current in zip(
        track[:-1],
        track[1:],
    ):
        delta_hours = (
            current.lead_time_hours
            - previous.lead_time_hours
        )

        if delta_hours <= 0:
            raise ValueError(
                "Track lead times must be strictly increasing."
            )

        distance = float(
            great_circle_distance_km(
                current.latitude,
                current.longitude,
                previous.latitude,
                previous.longitude,
            )
        )

        speed = distance / delta_hours

        bearing = initial_bearing_degrees(
            previous.latitude,
            previous.longitude,
            current.latitude,
            current.longitude,
        )

        cumulative_distance += distance

        result.append(
            TrackMotion(
                lead_time_hours=current.lead_time_hours,
                distance_km=distance,
                translation_speed_kmh=speed,
                bearing_degrees=bearing,
                cumulative_distance_km=cumulative_distance,
            )
        )

    return result
