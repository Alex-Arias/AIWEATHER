"""
Tropical cyclone tracker evaluation utilities.

Provides higher-level helpers for comparing native AIWeather
tropical cyclone tracks with candidate tracks produced by
Earth2Studio tracking algorithms.
"""

from __future__ import annotations

from dataclasses import dataclass

from .comparison import TrackComparison
from .earth2studio import Earth2StudioTrack
from .records import TrackRecord
from .selection import (
    TrackMatch,
    select_matching_track,
)


@dataclass(slots=True)
class TrackerEvaluation:
    """
    Evaluation of external tropical cyclone trackers against
    a reference AIWeather track.

    Parameters
    ----------
    reference_records
        Reference tropical cyclone track.

    wuduan_match
        Best matching Wu-Duan Earth2Studio track.

    vitart_match
        Best matching Vitart Earth2Studio track.
    """

    reference_records: list[TrackRecord]
    wuduan_match: TrackMatch | None
    vitart_match: TrackMatch | None

    @property
    def comparisons(
        self,
    ) -> dict[str, TrackComparison]:
        """
        Return available tracker comparisons.
        """
        result: dict[str, TrackComparison] = {}

        if self.wuduan_match is not None:
            result["wuduan"] = (
                self.wuduan_match.comparison
            )

        if self.vitart_match is not None:
            result["vitart"] = (
                self.vitart_match.comparison
            )

        return result


def compare_tracker_ensemble(
    reference_records: list[TrackRecord],
    *,
    initialization_time,
    wuduan_tracks: list[Earth2StudioTrack] | None = None,
    vitart_tracks: list[Earth2StudioTrack] | None = None,
    minimum_overlap: int = 1,
) -> TrackerEvaluation:
    """
    Compare Earth2Studio tracker ensembles with a reference track.

    Candidate tracks from each tracker are matched independently
    against the reference track. The best candidate is selected
    using the existing ``select_matching_track`` algorithm.

    Parameters
    ----------
    reference_records
        Reference AIWeather tropical cyclone track.

    initialization_time
        Forecast initialization time used when converting
        Earth2Studio tracks to TrackRecord objects.

    wuduan_tracks
        Candidate tracks returned by the Wu-Duan tracker.

    vitart_tracks
        Candidate tracks returned by the Vitart tracker.

    minimum_overlap
        Minimum number of common lead times required for a
        candidate track to be considered.

    Returns
    -------
    TrackerEvaluation
        Best matching track from each available tracker.
    """
    if not isinstance(
        reference_records,
        list,
    ):
        raise TypeError(
            "reference_records must be a list."
        )

    for record in reference_records:
        if not isinstance(
            record,
            TrackRecord,
        ):
            raise TypeError(
                "reference_records must contain "
                "TrackRecord objects."
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

    if (
        wuduan_tracks is not None
        and not isinstance(
            wuduan_tracks,
            list,
        )
    ):
        raise TypeError(
            "wuduan_tracks must be a list or None."
        )

    if (
        vitart_tracks is not None
        and not isinstance(
            vitart_tracks,
            list,
        )
    ):
        raise TypeError(
            "vitart_tracks must be a list or None."
        )

    wuduan_match = None

    if wuduan_tracks is not None:
        wuduan_match = select_matching_track(
            reference_records,
            wuduan_tracks,
            initialization_time=initialization_time,
            minimum_overlap=minimum_overlap,
            reference_name="aiweather",
            candidate_name="wuduan",
        )

    vitart_match = None

    if vitart_tracks is not None:
        vitart_match = select_matching_track(
            reference_records,
            vitart_tracks,
            initialization_time=initialization_time,
            minimum_overlap=minimum_overlap,
            reference_name="aiweather",
            candidate_name="vitart",
        )

    return TrackerEvaluation(
        reference_records=reference_records,
        wuduan_match=wuduan_match,
        vitart_match=vitart_match,
    )