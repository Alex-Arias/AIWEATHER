"""
Verification utilities for frozen operational tropical-cyclone tracks.

Operational tracks are verified only after tracking has been completed.
The verifying best track therefore cannot influence forecast-track
selection.
"""

from __future__ import annotations

from pathlib import Path

from aiweather.tracking import (
    TrackRecord,
    read_track_records_csv,
)

from aiweather.verification.comparison import (
    TrackVerification,
    compare_forecast_to_best_track,
)

from aiweather.verification.best_track import (
    best_track_to_records,
)
from aiweather.verification.ibtracs import (
    read_ibtracs_csv,
)

_PRESSURE_UNITS = {
    "pa",
    "pascal",
    "pascals",
}

_WIND_UNITS = {
    "m/s",
    "m s-1",
    "m s^-1",
    "ms-1",
}


def validate_operational_track_units(
    records: list[TrackRecord],
) -> None:
    """
    Validate units in a frozen operational AIWeather track.

    Legacy operational track CSV files may contain blank unit metadata.
    Their numerical pressure and maximum-wind values use the AIWeather
    operational convention of Pa and m/s, respectively.

    Explicit unit metadata must be compatible with that convention.
    No numerical conversion is performed.
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

        if record.pressure_units is not None:
            pressure_units = (
                str(record.pressure_units)
                .strip()
                .lower()
            )

            if pressure_units not in _PRESSURE_UNITS:
                raise ValueError(
                    "Operational track pressure units must be Pa "
                    "or blank legacy metadata."
                )

        if (
            record.max_wind is not None
            and record.wind_units is not None
        ):
            wind_units = (
                str(record.wind_units)
                .strip()
                .lower()
            )

            if wind_units not in _WIND_UNITS:
                raise ValueError(
                    "Operational track wind units must be m/s "
                    "or blank legacy metadata."
                )


def verify_operational_track(
    forecast_records: list[TrackRecord],
    observation_records: list[TrackRecord],
    *,
    forecast_name: str = "operational_forecast",
    observation_name: str = "best_track",
) -> TrackVerification:
    """
    Verify one frozen operational tropical-cyclone track.

    The forecast track must already have been generated and frozen
    independently of the verifying best track. No tracking, candidate
    association, or reference-informed track selection is performed
    here.

    Parameters
    ----------
    forecast_records
        Frozen operational AIWeather track records.

    observation_records
        Best-track records at observed valid times.

    forecast_name
        Label used in the returned verification result.

    observation_name
        Label used for the verifying reference track.

    Returns
    -------
    TrackVerification
        Exact-valid-time verification result.
    """
    validate_operational_track_units(
        forecast_records
    )

    return compare_forecast_to_best_track(
        forecast_records,
        observation_records,
        forecast_name=forecast_name,
        observation_name=observation_name,
    )


def verify_operational_track_csv(
    forecast_path: str | Path,
    ibtracs_path: str | Path,
    *,
    sid: str,
    initialization_time,
    forecast_name: str = "operational_forecast",
    observation_name: str = "IBTrACS",
    wind_column: str = "USA_WIND",
    pressure_column: str = "USA_PRES",
) -> TrackVerification:
    """
    Verify one frozen operational track CSV against IBTrACS.

    The forecast track is read exactly as previously exported.
    No tracking, candidate association, or reference-informed
    track selection is performed.

    Parameters
    ----------
    forecast_path
        Frozen operational AIWeather track CSV.

    ibtracs_path
        IBTrACS CSV file.

    sid
        IBTrACS storm identifier.

    initialization_time
        Forecast initialization time used to calculate best-track
        lead times.

    forecast_name
        Forecast/model label.

    observation_name
        Reference-track label.

    wind_column
        IBTrACS maximum-wind column.

    pressure_column
        IBTrACS central-pressure column.

    Returns
    -------
    TrackVerification
        Exact-valid-time verification result.
    """
    forecast_records = read_track_records_csv(
        forecast_path
    )

    best_track_points = read_ibtracs_csv(
        ibtracs_path,
        sid=sid,
        wind_column=wind_column,
        pressure_column=pressure_column,
    )

    observation_records = best_track_to_records(
        best_track_points,
        initialization_time=initialization_time,
    )

    return verify_operational_track(
        forecast_records,
        observation_records,
        forecast_name=forecast_name,
        observation_name=observation_name,
    )
