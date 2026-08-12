"""
Tropical cyclone track matching utilities.

Provides helpers for selecting the candidate track that best matches
a reference tropical cyclone track.
"""

from __future__ import annotations

from dataclasses import dataclass

from .comparison import (
    TrackComparison,
    compare_tracks,
)
from .earth2studio import (
    Earth2StudioTrack,
    earth2studio_track_to_records,
)
from .records import TrackRecord


@dataclass(slots=True)
class TrackMatch:
    """
    Result of matching a candidate track to a reference track.
    """

    track: Earth2StudioTrack
    records: list[TrackRecord]
    comparison: TrackComparison

    @property
    def path_id(self) -> int:
        """Earth2Studio path identifier."""
        return self.track.path_id

    @property
    def overlap_count(self) -> int:
        """Number of common lead times with the reference."""
        return self.comparison.overlap_count

    @property
    def mean_track_error_km(self) -> float:
        """Mean positional error relative to the reference."""
        return self.comparison.mean_track_error_km


def select_matching_track(
    reference_records: list[TrackRecord],
    candidate_tracks: list[Earth2StudioTrack],
    *,
    initialization_time,
    minimum_overlap: int = 1,
    reference_name: str = "reference",
    candidate_name: str = "candidate",
) -> TrackMatch | None:
    """
    Select the candidate track that best matches a reference track.

    Each candidate is converted to AIWeather TrackRecord objects and
    compared with the reference at common forecast lead times.

    Candidates with fewer than ``minimum_overlap`` common lead times
    are rejected. Among the remaining candidates, the candidate with
    the smallest mean positional error is selected.

    Parameters
    ----------
    reference_records
        Reference tropical cyclone track.

    candidate_tracks
        Candidate Earth2Studio tracks.

    initialization_time
        Forecast initialization time used when converting candidate
        tracks to TrackRecord objects.

    minimum_overlap
        Minimum number of common lead times required for a candidate
        to be considered.

    reference_name
        Label used for the reference track in TrackComparison.

    candidate_name
        Base label used for candidate tracks.

    Returns
    -------
    TrackMatch or None
        Best matching track, or None when no candidate satisfies the
        overlap requirement.
    """

    if not isinstance(
        reference_records,
        list,
    ):
        raise TypeError(
            "reference_records must be a list."
        )

    if not isinstance(
        candidate_tracks,
        list,
    ):
        raise TypeError(
            "candidate_tracks must be a list."
        )

    if not isinstance(
        minimum_overlap,
        int,
    ):
        raise TypeError(
            "minimum_overlap must be an integer."
        )

    if minimum_overlap < 1:
        raise ValueError(
            "minimum_overlap must be at least 1."
        )

    best_match = None

    for candidate in candidate_tracks:

        if not isinstance(
            candidate,
            Earth2StudioTrack,
        ):
            raise TypeError(
                "candidate_tracks must contain "
                "Earth2StudioTrack objects."
            )

        records = earth2studio_track_to_records(
            candidate,
            initialization_time=initialization_time,
        )

        comparison = compare_tracks(
            reference_records,
            records,
            tracker_a=reference_name,
            tracker_b=(
                f"{candidate_name}_"
                f"{candidate.path_id}"
            ),
        )

        if (
            comparison.overlap_count
            < minimum_overlap
        ):
            continue

        match = TrackMatch(
            track=candidate,
            records=records,
            comparison=comparison,
        )

        if best_match is None:
            best_match = match
            continue

        current_error = (
            match.mean_track_error_km
        )

        best_error = (
            best_match.mean_track_error_km
        )

        if current_error < best_error:
            best_match = match
            continue

        # Deterministic tie breaking:
        # prefer greater temporal overlap.
        if (
            current_error == best_error
            and match.overlap_count
            > best_match.overlap_count
        ):
            best_match = match
            continue

        # Final deterministic tie breaker.
        if (
            current_error == best_error
            and match.overlap_count
            == best_match.overlap_count
            and match.path_id
            < best_match.path_id
        ):
            best_match = match

    return best_match