"""
Forecast-to-best-track verification utilities.

Provides valid-time alignment and verification metrics for tropical
cyclone forecast tracks and observed best-track records.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from aiweather.tracking.records import TrackRecord
from aiweather.tracking.tropical_cyclone import (
    great_circle_distance_km,
)


@dataclass(slots=True)
class TrackVerification:
    """
    Forecast verification against an observed best track.
    """

    forecast_name: str
    observation_name: str
    table: pd.DataFrame

    @property
    def overlap_count(self) -> int:
        """Number of common valid times."""
        return len(self.table)

    @property
    def mean_track_error_km(self) -> float:
        """Mean forecast center-position error."""
        if self.table.empty:
            return float("nan")

        return float(
            self.table["track_error_km"].mean()
        )

    @property
    def rmse_track_error_km(self) -> float:
        """RMS forecast center-position error."""
        if self.table.empty:
            return float("nan")

        values = self.table[
            "track_error_km"
        ].to_numpy(dtype=float)

        return float(
            np.sqrt(
                np.mean(values**2)
            )
        )

    @property
    def median_track_error_km(self) -> float:
        """Median forecast center-position error."""
        if self.table.empty:
            return float("nan")

        return float(
            self.table[
                "track_error_km"
            ].median()
        )

    @property
    def maximum_track_error_km(self) -> float:
        """Maximum forecast center-position error."""
        if self.table.empty:
            return float("nan")

        return float(
            self.table[
                "track_error_km"
            ].max()
        )

    @property
    def mean_pressure_error_pa(self) -> float:
        """
        Mean signed forecast-minus-observation pressure error.
        """
        if self.table.empty:
            return float("nan")

        values = self.table[
            "pressure_error_pa"
        ].dropna()

        if values.empty:
            return float("nan")

        return float(values.mean())

    @property
    def mean_absolute_pressure_error_pa(
        self,
    ) -> float:
        """
        Mean absolute pressure error.
        """
        if self.table.empty:
            return float("nan")

        values = self.table[
            "pressure_error_pa"
        ].dropna()

        if values.empty:
            return float("nan")

        return float(
            values.abs().mean()
        )

    @property
    def rmse_pressure_error_pa(self) -> float:
        """
        RMS pressure error.
        """
        if self.table.empty:
            return float("nan")

        values = self.table[
            "pressure_error_pa"
        ].dropna().to_numpy(dtype=float)

        if len(values) == 0:
            return float("nan")

        return float(
            np.sqrt(
                np.mean(values**2)
            )
        )

    @property
    def mean_wind_error_ms(self) -> float:
        """
        Mean signed forecast-minus-observation wind error.
        """
        if self.table.empty:
            return float("nan")

        values = self.table[
            "wind_error_ms"
        ].dropna()

        if values.empty:
            return float("nan")

        return float(values.mean())

    @property
    def mean_absolute_wind_error_ms(
        self,
    ) -> float:
        """
        Mean absolute maximum-wind error.
        """
        if self.table.empty:
            return float("nan")

        values = self.table[
            "wind_error_ms"
        ].dropna()

        if values.empty:
            return float("nan")

        return float(
            values.abs().mean()
        )

    @property
    def rmse_wind_error_ms(self) -> float:
        """
        RMS maximum-wind error.
        """
        if self.table.empty:
            return float("nan")

        values = self.table[
            "wind_error_ms"
        ].dropna().to_numpy(dtype=float)

        if len(values) == 0:
            return float("nan")

        return float(
            np.sqrt(
                np.mean(values**2)
            )
        )


def align_by_valid_time(
    forecast_records: list[TrackRecord],
    observation_records: list[TrackRecord],
) -> list[
    tuple[
        TrackRecord,
        TrackRecord,
    ]
]:
    """
    Align forecast and best-track records using exact valid time.

    This allows observation records to have a finer sampling interval
    than the forecast. Only exact common valid times are retained.
    """

    if not isinstance(
        forecast_records,
        list,
    ):
        raise TypeError(
            "forecast_records must be a list."
        )

    if not isinstance(
        observation_records,
        list,
    ):
        raise TypeError(
            "observation_records must be a list."
        )

    for record in forecast_records:
        if not isinstance(
            record,
            TrackRecord,
        ):
            raise TypeError(
                "forecast_records must contain "
                "TrackRecord objects."
            )

    for record in observation_records:
        if not isinstance(
            record,
            TrackRecord,
        ):
            raise TypeError(
                "observation_records must contain "
                "TrackRecord objects."
            )

    forecast_by_time = {
        np.datetime64(
            record.valid_time
        ): record
        for record in forecast_records
    }

    observation_by_time = {
        np.datetime64(
            record.valid_time
        ): record
        for record in observation_records
    }

    common_times = sorted(
        set(forecast_by_time)
        & set(observation_by_time)
    )

    return [
        (
            forecast_by_time[valid_time],
            observation_by_time[valid_time],
        )
        for valid_time in common_times
    ]


def compare_forecast_to_best_track(
    forecast_records: list[TrackRecord],
    observation_records: list[TrackRecord],
    *,
    forecast_name: str = "forecast",
    observation_name: str = "best_track",
) -> TrackVerification:
    """
    Verify a tropical cyclone forecast against best-track observations.

    Position, central pressure, and maximum-wind errors are calculated
    at exact common valid times.

    Error convention
    ----------------
    Intensity errors are defined as:

        forecast - observation

    Therefore:

    * positive pressure error means the forecast cyclone is weaker
      in central pressure;
    * negative wind error means the forecast cyclone has weaker
      maximum winds.
    """

    aligned = align_by_valid_time(
        forecast_records,
        observation_records,
    )

    rows = []

    for (
        forecast,
        observation,
    ) in aligned:

        track_error = (
            great_circle_distance_km(
                forecast.latitude,
                forecast.longitude,
                observation.latitude,
                observation.longitude,
            )
        )

        if (
            forecast.pressure is not None
            and observation.pressure is not None
        ):
            pressure_error = (
                float(forecast.pressure)
                - float(observation.pressure)
            )
        else:
            pressure_error = np.nan

        if (
            forecast.max_wind is not None
            and observation.max_wind is not None
        ):
            wind_error = (
                float(forecast.max_wind)
                - float(observation.max_wind)
            )
        else:
            wind_error = np.nan

        rows.append(
            {
                "valid_time":
                    np.datetime64(
                        forecast.valid_time
                    ),
                "lead_time_hours":
                    int(
                        forecast.lead_time_hours
                    ),

                "forecast_latitude":
                    float(
                        forecast.latitude
                    ),
                "forecast_longitude":
                    float(
                        forecast.longitude
                    ),

                "observed_latitude":
                    float(
                        observation.latitude
                    ),
                "observed_longitude":
                    float(
                        observation.longitude
                    ),

                "track_error_km":
                    float(track_error),

                "forecast_pressure_pa":
                    forecast.pressure,
                "observed_pressure_pa":
                    observation.pressure,
                "pressure_error_pa":
                    pressure_error,

                "forecast_wind_ms":
                    forecast.max_wind,
                "observed_wind_ms":
                    observation.max_wind,
                "wind_error_ms":
                    wind_error,
            }
        )

    table = pd.DataFrame(rows)

    return TrackVerification(
        forecast_name=forecast_name,
        observation_name=observation_name,
        table=table,
    )