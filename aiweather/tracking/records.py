"""
Track-record utilities for AIWeather.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime

import numpy as np
import pandas as pd
import xarray as xr

from .motion import track_motion
from .tropical_cyclone import TrackPoint


@dataclass(frozen=True, slots=True)
class TrackRecord:
    """
    One complete forecast-track record.

    Attributes
    ----------
    lead_time_hours : int
        Forecast lead time in hours.

    valid_time : np.datetime64
        Forecast valid time.

    latitude : float
        Track-center latitude in degrees.

    longitude : float
        Track-center longitude in degrees.

    pressure : float
        Minimum pressure at the tracked center.

    pressure_units : str or None
        Pressure units when available.

    max_wind : float or None
        Maximum local wind speed around the center.

    wind_units : str or None
        Wind-speed units when available.

    distance_km : float or None
        Distance traveled since the previous track point.

    translation_speed_kmh : float or None
        Translation speed since the previous track point.

    bearing_degrees : float or None
        Initial bearing from the previous point, clockwise
        from north.

    cumulative_distance_km : float
        Total distance traveled from the beginning of the track.
    """

    lead_time_hours: int
    valid_time: np.datetime64

    latitude: float
    longitude: float

    pressure: float
    pressure_units: str | None = None

    max_wind: float | None = None
    wind_units: str | None = None

    distance_km: float | None = None
    translation_speed_kmh: float | None = None
    bearing_degrees: float | None = None

    cumulative_distance_km: float = 0.0


def _normalize_initialization_time(
    initialization_time: datetime | np.datetime64 | str,
) -> np.datetime64:
    """
    Convert a supported initialization-time value to np.datetime64.
    """
    try:
        value = np.datetime64(
            initialization_time,
            "ns",
        )
    except Exception as exc:
        raise ValueError(
            f"Invalid initialization_time: {initialization_time!r}"
        ) from exc

    if np.isnat(value):
        raise ValueError(
            "initialization_time cannot be NaT."
        )

    return value


def build_track_records(
    track: list[TrackPoint],
    *,
    initialization_time: datetime | np.datetime64 | str,
) -> list[TrackRecord]:
    """
    Build complete track records from TrackPoint objects.

    Parameters
    ----------
    track : list[TrackPoint]
        Candidate cyclone-center track.

    initialization_time : datetime, np.datetime64, or str
        Forecast initialization time.

    Returns
    -------
    list[TrackRecord]
        One record for every TrackPoint.

    Raises
    ------
    TypeError
        If track is not a list or contains non-TrackPoint objects.

    ValueError
        If initialization_time is invalid or track lead times
        are not strictly increasing.
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

    if not track:
        return []

    initialization = _normalize_initialization_time(
        initialization_time
    )

    motion = track_motion(track)

    # Motion records correspond to destination points, so index
    # them by destination lead time.
    motion_by_lead = {
        item.lead_time_hours: item
        for item in motion
    }

    records: list[TrackRecord] = []

    for point in track:
        valid_time = (
            initialization
            + np.timedelta64(
                point.lead_time_hours,
                "h",
            )
        )

        motion_item = motion_by_lead.get(
            point.lead_time_hours
        )

        if motion_item is None:
            distance_km = None
            translation_speed_kmh = None
            bearing_degrees = None
            cumulative_distance_km = 0.0
        else:
            distance_km = motion_item.distance_km
            translation_speed_kmh = (
                motion_item.translation_speed_kmh
            )
            bearing_degrees = (
                motion_item.bearing_degrees
            )
            cumulative_distance_km = (
                motion_item.cumulative_distance_km
            )

        records.append(
            TrackRecord(
                lead_time_hours=point.lead_time_hours,
                valid_time=valid_time,
                latitude=point.latitude,
                longitude=point.longitude,
                pressure=point.pressure,
                pressure_units=point.pressure_units,
                max_wind=point.max_wind,
                wind_units=point.wind_units,
                distance_km=distance_km,
                translation_speed_kmh=translation_speed_kmh,
                bearing_degrees=bearing_degrees,
                cumulative_distance_km=cumulative_distance_km,
            )
        )

    return records


def dataframe_to_records(
    dataframe: pd.DataFrame,
) -> list[TrackRecord]:
    """
    Convert a pandas DataFrame to TrackRecord objects.

    Parameters
    ----------
    dataframe : pandas.DataFrame
        Tabular track representation using the canonical
        TrackRecord column names.

    Returns
    -------
    list[TrackRecord]
        Track records ordered as supplied in the dataframe.
    """
    if not isinstance(dataframe, pd.DataFrame):
        raise TypeError(
            "dataframe must be a pandas DataFrame."
        )

    required = {
        "lead_time_hours",
        "valid_time",
        "latitude",
        "longitude",
        "pressure",
    }

    missing = required.difference(
        dataframe.columns
    )

    if missing:
        raise ValueError(
            "Track dataframe is missing required columns: "
            + ", ".join(sorted(missing))
        )

    records: list[TrackRecord] = []

    for _, row in dataframe.iterrows():
        valid_time = np.datetime64(
            pd.to_datetime(
                row["valid_time"]
            ),
            "ns",
        )

        if np.isnat(valid_time):
            raise ValueError(
                "Track valid_time cannot be NaT."
            )

        def optional_value(name):
            if name not in dataframe.columns:
                return None

            value = row[name]

            if pd.isna(value):
                return None

            return value

        records.append(
            TrackRecord(
                lead_time_hours=int(
                    row["lead_time_hours"]
                ),
                valid_time=valid_time,
                latitude=float(
                    row["latitude"]
                ),
                longitude=float(
                    row["longitude"]
                ),
                pressure=float(
                    row["pressure"]
                ),
                pressure_units=optional_value(
                    "pressure_units"
                ),
                max_wind=(
                    None
                    if optional_value("max_wind") is None
                    else float(
                        optional_value("max_wind")
                    )
                ),
                wind_units=optional_value(
                    "wind_units"
                ),
                distance_km=(
                    None
                    if optional_value("distance_km") is None
                    else float(
                        optional_value("distance_km")
                    )
                ),
                translation_speed_kmh=(
                    None
                    if optional_value(
                        "translation_speed_kmh"
                    ) is None
                    else float(
                        optional_value(
                            "translation_speed_kmh"
                        )
                    )
                ),
                bearing_degrees=(
                    None
                    if optional_value(
                        "bearing_degrees"
                    ) is None
                    else float(
                        optional_value(
                            "bearing_degrees"
                        )
                    )
                ),
                cumulative_distance_km=(
                    0.0
                    if optional_value(
                        "cumulative_distance_km"
                    ) is None
                    else float(
                        optional_value(
                            "cumulative_distance_km"
                        )
                    )
                ),
            )
        )

    return records


def read_track_records_csv(
    path,
) -> list[TrackRecord]:
    """
    Read canonical TrackRecord objects from a CSV file.
    """
    dataframe = pd.read_csv(
        path
    )

    return dataframe_to_records(
        dataframe
    )


def records_to_dataframe(
    records: list[TrackRecord],
) -> pd.DataFrame:
    """
    Convert TrackRecord objects to a pandas DataFrame.

    Parameters
    ----------
    records : list[TrackRecord]
        Track records to convert.

    Returns
    -------
    pandas.DataFrame
        Tabular track representation.
    """
    if not isinstance(records, list):
        raise TypeError(
            "records must be a list of TrackRecord objects."
        )

    for record in records:
        if not isinstance(record, TrackRecord):
            raise TypeError(
                "records must contain only TrackRecord objects."
            )

    columns = [
        "lead_time_hours",
        "valid_time",
        "latitude",
        "longitude",
        "pressure",
        "pressure_units",
        "max_wind",
        "wind_units",
        "distance_km",
        "translation_speed_kmh",
        "bearing_degrees",
        "cumulative_distance_km",
    ]

    if not records:
        return pd.DataFrame(
            columns=columns
        )

    dataframe = pd.DataFrame(
        [
            asdict(record)
            for record in records
        ],
        columns=columns,
    )

    dataframe["valid_time"] = pd.to_datetime(
        dataframe["valid_time"]
    )

    return dataframe


def records_to_xarray(
    records: list[TrackRecord],
) -> xr.Dataset:
    """
    Convert TrackRecord objects to an xarray Dataset.

    Parameters
    ----------
    records : list[TrackRecord]
        Track records to convert.

    Returns
    -------
    xr.Dataset
        Track represented along a ``track_point`` dimension.
    """
    dataframe = records_to_dataframe(
        records
    )

    if dataframe.empty:
        return xr.Dataset(
            coords={
                "track_point": np.array(
                    [],
                    dtype=int,
                )
            }
        )

    track_point = np.arange(
        len(dataframe),
        dtype=int,
    )

    dataset = xr.Dataset(
        data_vars={
            "lead_time_hours": (
                "track_point",
                dataframe["lead_time_hours"].to_numpy(
                    dtype=int
                ),
            ),
            "latitude": (
                "track_point",
                dataframe["latitude"].to_numpy(
                    dtype=float
                ),
            ),
            "longitude": (
                "track_point",
                dataframe["longitude"].to_numpy(
                    dtype=float
                ),
            ),
            "pressure": (
                "track_point",
                dataframe["pressure"].to_numpy(
                    dtype=float
                ),
            ),
            "max_wind": (
                "track_point",
                dataframe["max_wind"].to_numpy(
                    dtype=float
                ),
            ),
            "distance_km": (
                "track_point",
                dataframe["distance_km"].to_numpy(
                    dtype=float
                ),
            ),
            "translation_speed_kmh": (
                "track_point",
                dataframe[
                    "translation_speed_kmh"
                ].to_numpy(
                    dtype=float
                ),
            ),
            "bearing_degrees": (
                "track_point",
                dataframe["bearing_degrees"].to_numpy(
                    dtype=float
                ),
            ),
            "cumulative_distance_km": (
                "track_point",
                dataframe[
                    "cumulative_distance_km"
                ].to_numpy(
                    dtype=float
                ),
            ),
        },
        coords={
            "track_point": track_point,
            "valid_time": (
                "track_point",
                dataframe["valid_time"].to_numpy(
                    dtype="datetime64[ns]"
                ),
            ),
        },
    )

    # Preserve units only when the complete track has a single
    # consistent non-null unit.
    pressure_units = {
        record.pressure_units
        for record in records
        if record.pressure_units is not None
    }

    if len(pressure_units) == 1:
        dataset["pressure"].attrs["units"] = (
            pressure_units.pop()
        )

    wind_units = {
        record.wind_units
        for record in records
        if record.wind_units is not None
    }

    if len(wind_units) == 1:
        dataset["max_wind"].attrs["units"] = (
            wind_units.pop()
        )

    dataset["distance_km"].attrs["units"] = "km"
    dataset["translation_speed_kmh"].attrs[
        "units"
    ] = "km h-1"
    dataset["bearing_degrees"].attrs[
        "units"
    ] = "degrees"
    dataset["cumulative_distance_km"].attrs[
        "units"
    ] = "km"

    return dataset
