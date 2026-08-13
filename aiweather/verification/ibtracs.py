"""
IBTrACS best-track ingestion utilities.

Reads IBTrACS CSV data and converts storm observations into
AIWeather BestTrackPoint objects.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .best_track import BestTrackPoint


KNOT_TO_MS = 0.514444
HPA_TO_PA = 100.0


def _optional_float(value):
    """
    Convert an IBTrACS CSV value to float or None.
    """
    if pd.isna(value):
        return None

    if isinstance(value, str):
        value = value.strip()

        if value == "":
            return None

    return float(value)


def read_ibtracs_csv(
    path: str | Path,
    *,
    sid: str,
    wind_column: str = "USA_WIND",
    pressure_column: str = "USA_PRES",
) -> list[BestTrackPoint]:
    """
    Read one storm from an IBTrACS CSV file.

    Parameters
    ----------
    path
        IBTrACS CSV file.

    sid
        IBTrACS storm identifier.

    wind_column
        Column containing maximum sustained wind in knots.

    pressure_column
        Column containing minimum central pressure in hPa.

    Returns
    -------
    list[BestTrackPoint]
        Storm observations ordered by valid time.
    """

    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"IBTrACS CSV file not found: {path}"
        )

    if not isinstance(
        sid,
        str,
    ):
        raise TypeError(
            "sid must be a string."
        )

    sid = sid.strip()

    if not sid:
        raise ValueError(
            "sid cannot be empty."
        )

    data = pd.read_csv(
        path,
        low_memory=False,
    )

    required = {
        "SID",
        "ISO_TIME",
        "LAT",
        "LON",
        wind_column,
        pressure_column,
    }

    missing = required.difference(
        data.columns
    )

    if missing:
        raise ValueError(
            "IBTrACS CSV is missing required columns: "
            + ", ".join(sorted(missing))
        )

    storm = data.loc[
        data["SID"].astype(str) == sid
    ].copy()

    if storm.empty:
        raise ValueError(
            f"IBTrACS storm SID {sid!r} was not found."
        )

    points = []

    for _, row in storm.iterrows():

        time_value = str(
            row["ISO_TIME"]
        ).strip()

        if not time_value:
            continue

        try:
            valid_time = np.datetime64(
                time_value
            )
        except ValueError as exc:
            raise ValueError(
                f"Invalid IBTrACS ISO_TIME: "
                f"{time_value!r}"
            ) from exc

        latitude = _optional_float(
            row["LAT"]
        )

        longitude = _optional_float(
            row["LON"]
        )

        if (
            latitude is None
            or longitude is None
        ):
            continue

        pressure_hpa = _optional_float(
            row[pressure_column]
        )

        wind_knots = _optional_float(
            row[wind_column]
        )

        pressure_pa = (
            None
            if pressure_hpa is None
            else pressure_hpa * HPA_TO_PA
        )

        wind_ms = (
            None
            if wind_knots is None
            else wind_knots * KNOT_TO_MS
        )

        points.append(
            BestTrackPoint(
                valid_time=valid_time,
                latitude=latitude,
                longitude=longitude,
                pressure=pressure_pa,
                max_wind=wind_ms,
                pressure_units=(
                    None
                    if pressure_pa is None
                    else "Pa"
                ),
                wind_units=(
                    None
                    if wind_ms is None
                    else "m/s"
                ),
            )
        )

    points.sort(
        key=lambda item: item.valid_time
    )

    return points