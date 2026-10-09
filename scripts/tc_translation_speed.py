"""Translation-speed diagnostics for operational cyclone tracks."""

import numpy as np
import pandas as pd


EARTH_RADIUS_KM = 6371.0


def calculate_translation_speed(track):
    """
    Calculate interval-average cyclone translation speed.

    Parameters
    ----------
    track : pandas.DataFrame
        Requires lead_time_hours, latitude, and longitude.

    Returns
    -------
    pandas.DataFrame
        Copy of the input with translation_speed_kmh.
        The first position has undefined translation speed.
    """
    required = {
        "lead_time_hours",
        "latitude",
        "longitude",
    }

    missing = required.difference(track.columns)

    if missing:
        raise ValueError(
            f"Missing required columns: {sorted(missing)}"
        )

    df = track.copy()

    if df.empty:
        df["translation_speed_kmh"] = pd.Series(dtype=float)
        return df

    lead = df["lead_time_hours"].to_numpy(dtype=float)
    lat = df["latitude"].to_numpy(dtype=float)
    lon = df["longitude"].to_numpy(dtype=float)

    if not (
        np.isfinite(lead).all()
        and np.isfinite(lat).all()
        and np.isfinite(lon).all()
    ):
        raise ValueError("Track coordinates and times must be finite")

    if np.any(np.diff(lead) <= 0):
        raise ValueError("Lead times must be strictly increasing")

    if np.any(np.abs(lat) > 90):
        raise ValueError("Latitude outside valid range")

    speed = np.full(len(df), np.nan)

    if len(df) > 1:
        lat_rad = np.radians(lat)

        dlat = np.diff(lat_rad)

        # Handles longitude coordinates in either
        # [-180, 180] or [0, 360] convention.
        dlon_deg = (
            np.diff(lon) + 180.0
        ) % 360.0 - 180.0

        dlon = np.radians(dlon_deg)

        a = (
            np.sin(dlat / 2.0) ** 2
            + np.cos(lat_rad[:-1])
            * np.cos(lat_rad[1:])
            * np.sin(dlon / 2.0) ** 2
        )

        distance_km = (
            2.0
            * EARTH_RADIUS_KM
            * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))
        )

        speed[1:] = distance_km / np.diff(lead)

    df["translation_speed_kmh"] = speed

    return df
