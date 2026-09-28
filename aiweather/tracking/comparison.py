"""
Tropical cyclone track comparison utilities.

Provides common structures and functions for aligning and comparing
tracks produced by different tracking algorithms.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .records import TrackRecord
from .tropical_cyclone import great_circle_distance_km


@dataclass(slots=True)
class TrackComparison:
    """
    Pairwise comparison between two tropical cyclone tracks.
    """

    tracker_a: str
    tracker_b: str
    table: pd.DataFrame

    @property
    def overlap_count(self) -> int:
        """Number of common forecast lead times."""
        return len(self.table)

    @property
    def mean_track_error_km(self) -> float:
        """Mean great-circle separation between tracks."""
        if self.table.empty:
            return float("nan")

        return float(
            self.table["track_error_km"].mean()
        )

    @property
    def rmse_track_error_km(self) -> float:
        """Root-mean-square great-circle separation."""
        if self.table.empty:
            return float("nan")

        values = self.table[
            "track_error_km"
        ].to_numpy(dtype=float)

        return float(
            np.sqrt(np.mean(values**2))
        )

    @property
    def median_track_error_km(self) -> float:
        """Median great-circle separation."""
        if self.table.empty:
            return float("nan")

        return float(
            self.table["track_error_km"].median()
        )

    @property
    def maximum_track_error_km(self) -> float:
        """Maximum great-circle separation."""
        if self.table.empty:
            return float("nan")

        return float(
            self.table["track_error_km"].max()
        )


def align_tracks(
    track_a: list[TrackRecord],
    track_b: list[TrackRecord],
) -> list[tuple[TrackRecord, TrackRecord]]:
    """
    Align two tracks using common forecast lead times.

    Parameters
    ----------
    track_a, track_b
        Tropical cyclone track records.

    Returns
    -------
    list of tuple
        Matching TrackRecord pairs ordered by lead time.
    """

    by_lead_a = {
        int(record.lead_time_hours): record
        for record in track_a
    }

    by_lead_b = {
        int(record.lead_time_hours): record
        for record in track_b
    }

    common_leads = sorted(
        set(by_lead_a) & set(by_lead_b)
    )

    return [
        (
            by_lead_a[lead],
            by_lead_b[lead],
        )
        for lead in common_leads
    ]



def align_tracks_by_valid_time(
    track_a: list[TrackRecord],
    track_b: list[TrackRecord],
) -> list[tuple[TrackRecord, TrackRecord]]:
    """
    Align two tracks using exact common valid times.

    This alignment is intended for comparisons between forecast
    cycles with different initialization times, where equal forecast
    leads do not represent the same physical time.
    """

    by_time_a = {
        np.datetime64(record.valid_time):
            record
        for record in track_a
    }

    by_time_b = {
        np.datetime64(record.valid_time):
            record
        for record in track_b
    }

    common_times = sorted(
        set(by_time_a)
        & set(by_time_b)
    )

    return [
        (
            by_time_a[valid_time],
            by_time_b[valid_time],
        )
        for valid_time in common_times
    ]


def compare_tracks_by_valid_time(
    track_a: list[TrackRecord],
    track_b: list[TrackRecord],
    *,
    tracker_a: str = "tracker_a",
    tracker_b: str = "tracker_b",
) -> TrackComparison:
    """
    Compare two tropical cyclone tracks at exact common valid times.

    Unlike ``compare_tracks``, this function is appropriate for
    comparisons between forecast cycles initialized at different
    times.
    """

    aligned = align_tracks_by_valid_time(
        track_a,
        track_b,
    )

    rows = []

    for record_a, record_b in aligned:
        distance = great_circle_distance_km(
            record_a.latitude,
            record_a.longitude,
            record_b.latitude,
            record_b.longitude,
        )

        if (
            record_a.pressure is not None
            and record_b.pressure is not None
        ):
            pressure_difference = (
                record_a.pressure
                - record_b.pressure
            )
        else:
            pressure_difference = np.nan

        if (
            record_a.max_wind is not None
            and record_b.max_wind is not None
        ):
            wind_difference = (
                record_a.max_wind
                - record_b.max_wind
            )
        else:
            wind_difference = np.nan

        rows.append(
            {
                "lead_time_hours":
                    record_b.lead_time_hours,
                "valid_time":
                    record_b.valid_time,
                "latitude_a":
                    record_a.latitude,
                "longitude_a":
                    record_a.longitude,
                "latitude_b":
                    record_b.latitude,
                "longitude_b":
                    record_b.longitude,
                "pressure_a":
                    record_a.pressure,
                "pressure_b":
                    record_b.pressure,
                "max_wind_a":
                    record_a.max_wind,
                "max_wind_b":
                    record_b.max_wind,
                "track_error_km":
                    distance,
                "pressure_difference":
                    pressure_difference,
                "wind_difference":
                    wind_difference,
            }
        )

    table = pd.DataFrame(rows)

    return TrackComparison(
        tracker_a=tracker_a,
        tracker_b=tracker_b,
        table=table,
    )


def compare_tracks(
    track_a: list[TrackRecord],
    track_b: list[TrackRecord],
    *,
    tracker_a: str = "tracker_a",
    tracker_b: str = "tracker_b",
) -> TrackComparison:
    """
    Compare two tropical cyclone tracks at common lead times.
    """

    aligned = align_tracks(
        track_a,
        track_b,
    )

    rows = []

    for record_a, record_b in aligned:
        distance = great_circle_distance_km(
            record_a.latitude,
            record_a.longitude,
            record_b.latitude,
            record_b.longitude,
        )

        if (
            record_a.pressure is not None
            and record_b.pressure is not None
        ):
            pressure_difference = (
                record_a.pressure
                - record_b.pressure
            )
        else:
            pressure_difference = np.nan

        if (
            record_a.max_wind is not None
            and record_b.max_wind is not None
        ):
            wind_difference = (
                record_a.max_wind
                - record_b.max_wind
            )
        else:
            wind_difference = np.nan

        rows.append(
            {
                "lead_time_hours":
                    record_a.lead_time_hours,
                "valid_time":
                    record_a.valid_time,
                "latitude_a":
                    record_a.latitude,
                "longitude_a":
                    record_a.longitude,
                "latitude_b":
                    record_b.latitude,
                "longitude_b":
                    record_b.longitude,
                "pressure_a":
                    record_a.pressure,
                "pressure_b":
                    record_b.pressure,
                "max_wind_a":
                    record_a.max_wind,
                "max_wind_b":
                    record_b.max_wind,
                "track_error_km":
                    distance,
                "pressure_difference":
                    pressure_difference,
                "wind_difference":
                    wind_difference,
            }
        )

    table = pd.DataFrame(rows)

    return TrackComparison(
        tracker_a=tracker_a,
        tracker_b=tracker_b,
        table=table,
    )
