"""
Temporal association of tropical-cyclone candidate centers.

This module links spatial candidate centers detected at consecutive
forecast lead times into candidate trajectories.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from aiweather.diagnostics import PressureMinimum

from .tropical_cyclone import great_circle_distance_km


@dataclass(frozen=True)
class CandidatePoint:
    """
    Pressure-minimum candidate at one forecast lead time.
    """

    lead_time_hours: int
    pressure: float
    latitude: float
    longitude: float


@dataclass(frozen=True)
class CandidateTrack:
    """
    Sequence of temporally associated candidate centers.
    """

    points: tuple[CandidatePoint, ...]

    def __len__(self) -> int:
        return len(self.points)

    @property
    def first(self) -> CandidatePoint:
        return self.points[0]

    @property
    def last(self) -> CandidatePoint:
        return self.points[-1]


def _make_candidate_point(
    lead_time_hours: int,
    minimum: PressureMinimum,
) -> CandidatePoint:
    return CandidatePoint(
        lead_time_hours=int(lead_time_hours),
        pressure=float(minimum.value),
        latitude=float(minimum.latitude),
        longitude=float(minimum.longitude),
    )


def associate_candidates(
    candidates_by_lead: Sequence[
        tuple[int, Sequence[PressureMinimum]]
    ],
    *,
    maximum_displacement_km: float = 500.0,
) -> list[CandidateTrack]:
    """
    Associate pressure-minimum candidates through forecast time.

    Candidates at each lead time are linked to existing trajectories
    using nearest-neighbor great-circle distance. A candidate can be
    associated only when it lies within ``maximum_displacement_km``
    of the final point in an existing trajectory.

    Each trajectory can receive at most one candidate at a given
    lead time, and each candidate can be assigned to at most one
    trajectory. Unmatched candidates begin new trajectories.

    Parameters
    ----------
    candidates_by_lead
        Sequence of ``(lead_time_hours, candidates)`` pairs.
        Candidates should normally come from
        ``detect_pressure_minima``.

    maximum_displacement_km
        Maximum distance allowed between consecutive associated
        candidate centers.

    Returns
    -------
    list[CandidateTrack]
        Candidate trajectories.

    Raises
    ------
    ValueError
        If maximum displacement is not positive, lead times are not
        strictly increasing, or duplicate lead times are supplied.
    """
    if maximum_displacement_km <= 0.0:
        raise ValueError(
            "maximum_displacement_km must be greater than zero."
        )

    if not candidates_by_lead:
        return []

    lead_times = [
        int(lead_time)
        for lead_time, _ in candidates_by_lead
    ]

    if any(
        current <= previous
        for previous, current in zip(
            lead_times[:-1],
            lead_times[1:],
        )
    ):
        raise ValueError(
            "Lead times must be strictly increasing."
        )

    tracks: list[list[CandidatePoint]] = []

    for lead_time_hours, minima in candidates_by_lead:
        current_points = [
            _make_candidate_point(
                lead_time_hours,
                minimum,
            )
            for minimum in minima
        ]

        if not tracks:
            tracks.extend(
                [[point] for point in current_points]
            )
            continue

        # Construct all allowed track-candidate pairings.
        possible_matches = []

        for track_index, track in enumerate(tracks):
            previous = track[-1]

            for candidate_index, candidate in enumerate(
                current_points
            ):
                distance_km = float(
                    great_circle_distance_km(
                        previous.latitude,
                        previous.longitude,
                        candidate.latitude,
                        candidate.longitude,
                    )
                )

                if distance_km <= maximum_displacement_km:
                    possible_matches.append(
                        (
                            distance_km,
                            track_index,
                            candidate_index,
                        )
                    )

        # Greedy nearest-neighbor assignment.
        possible_matches.sort(
            key=lambda match: match[0]
        )

        assigned_tracks: set[int] = set()
        assigned_candidates: set[int] = set()

        for (
            _distance_km,
            track_index,
            candidate_index,
        ) in possible_matches:
            if track_index in assigned_tracks:
                continue

            if candidate_index in assigned_candidates:
                continue

            tracks[track_index].append(
                current_points[candidate_index]
            )

            assigned_tracks.add(track_index)
            assigned_candidates.add(candidate_index)

        # Candidates with no compatible predecessor start new tracks.
        for candidate_index, candidate in enumerate(
            current_points
        ):
            if candidate_index not in assigned_candidates:
                tracks.append([candidate])

    return [
        CandidateTrack(
            points=tuple(track)
        )
        for track in tracks
    ]
