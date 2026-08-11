"""
Tropical-cyclone candidate classification for AIWeather.

This module classifies already-tracked candidate centers using
configurable pressure, wind, and persistence criteria.

The classification is intentionally separate from the tracking
algorithm. A tracked pressure minimum is not automatically assumed
to be a tropical cyclone.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .records import TrackRecord


@dataclass(frozen=True, slots=True)
class TCClassification:
    """
    Classification of one forecast track point.

    Attributes
    ----------
    lead_time_hours : int
        Forecast lead time in hours.

    valid_time : np.datetime64
        Forecast valid time.

    wind_threshold_met : bool
        Whether the configured wind criterion is satisfied.

    pressure_threshold_met : bool
        Whether the configured pressure criterion is satisfied.

    intensity_criteria_met : bool
        Whether the enabled intensity criteria are satisfied.

    persistence_met : bool
        Whether the point belongs to a sufficiently persistent
        sequence of intensity-qualified points.

    is_candidate : bool
        Final TC-candidate flag.
    """

    lead_time_hours: int
    valid_time: np.datetime64

    wind_threshold_met: bool
    pressure_threshold_met: bool

    intensity_criteria_met: bool
    persistence_met: bool

    is_candidate: bool


def _validate_records(
    records: list[TrackRecord],
) -> None:
    """
    Validate a sequence of TrackRecord objects.
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

    for previous, current in zip(
        records[:-1],
        records[1:],
    ):
        if (
            current.lead_time_hours
            <= previous.lead_time_hours
        ):
            raise ValueError(
                "Track-record lead times must be strictly increasing."
            )


def _persistent_mask(
    qualified: list[bool],
    minimum_consecutive_points: int,
) -> list[bool]:
    """
    Mark points belonging to persistent qualified sequences.
    """
    persistent = [
        False
        for _ in qualified
    ]

    start = None

    for index, value in enumerate(
        qualified + [False]
    ):
        if value and start is None:
            start = index

        elif not value and start is not None:
            run_length = index - start

            if run_length >= minimum_consecutive_points:
                for run_index in range(
                    start,
                    index,
                ):
                    persistent[run_index] = True

            start = None

    return persistent


def classify_track(
    records: list[TrackRecord],
    *,
    minimum_wind: float | None = None,
    maximum_pressure: float | None = None,
    minimum_consecutive_points: int = 3,
    require_all_enabled_criteria: bool = True,
) -> list[TCClassification]:
    """
    Classify a tracked system as a persistent TC candidate.

    Parameters
    ----------
    records : list[TrackRecord]
        Track records to classify.

    minimum_wind : float, optional
        Minimum local maximum wind required to satisfy the wind
        criterion.

        The caller is responsible for ensuring the threshold uses
        the same units as ``TrackRecord.max_wind``.

    maximum_pressure : float, optional
        Maximum center pressure allowed to satisfy the pressure
        criterion.

        The caller is responsible for ensuring the threshold uses
        the same units as ``TrackRecord.pressure``.

    minimum_consecutive_points : int, default=3
        Minimum number of consecutive qualifying forecast points
        required for persistence.

    require_all_enabled_criteria : bool, default=True
        If True, all enabled intensity criteria must be satisfied.
        If False, at least one enabled intensity criterion must be
        satisfied.

    Returns
    -------
    list[TCClassification]
        One classification object for every TrackRecord.

    Raises
    ------
    TypeError
        If records is not a valid TrackRecord list.

    ValueError
        If no intensity criteria are enabled, thresholds are invalid,
        or minimum_consecutive_points is less than one.
    """
    _validate_records(
        records
    )

    if minimum_wind is None and maximum_pressure is None:
        raise ValueError(
            "At least one intensity criterion must be enabled."
        )

    if minimum_wind is not None and minimum_wind < 0.0:
        raise ValueError(
            "minimum_wind must be greater than or equal to zero."
        )

    if (
        maximum_pressure is not None
        and not np.isfinite(maximum_pressure)
    ):
        raise ValueError(
            "maximum_pressure must be finite."
        )

    if minimum_consecutive_points < 1:
        raise ValueError(
            "minimum_consecutive_points must be at least 1."
        )

    if not records:
        return []

    wind_flags: list[bool] = []
    pressure_flags: list[bool] = []
    intensity_flags: list[bool] = []

    for record in records:
        if minimum_wind is None:
            wind_met = True
        else:
            wind_met = (
                record.max_wind is not None
                and np.isfinite(record.max_wind)
                and record.max_wind >= minimum_wind
            )

        if maximum_pressure is None:
            pressure_met = True
        else:
            pressure_met = (
                np.isfinite(record.pressure)
                and record.pressure <= maximum_pressure
            )

        wind_flags.append(
            bool(wind_met)
        )

        pressure_flags.append(
            bool(pressure_met)
        )

        enabled_results = []

        if minimum_wind is not None:
            enabled_results.append(
                bool(wind_met)
            )

        if maximum_pressure is not None:
            enabled_results.append(
                bool(pressure_met)
            )

        if require_all_enabled_criteria:
            intensity_met = all(
                enabled_results
            )
        else:
            intensity_met = any(
                enabled_results
            )

        intensity_flags.append(
            bool(intensity_met)
        )

    persistence_flags = _persistent_mask(
        intensity_flags,
        minimum_consecutive_points,
    )

    classifications: list[TCClassification] = []

    for (
        record,
        wind_met,
        pressure_met,
        intensity_met,
        persistence_met,
    ) in zip(
        records,
        wind_flags,
        pressure_flags,
        intensity_flags,
        persistence_flags,
    ):
        classifications.append(
            TCClassification(
                lead_time_hours=record.lead_time_hours,
                valid_time=record.valid_time,
                wind_threshold_met=wind_met,
                pressure_threshold_met=pressure_met,
                intensity_criteria_met=intensity_met,
                persistence_met=persistence_met,
                is_candidate=(
                    intensity_met
                    and persistence_met
                ),
            )
        )

    return classifications
