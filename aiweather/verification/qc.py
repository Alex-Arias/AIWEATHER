"""
Quality control for tropical cyclone tracker associations.

QC is intentionally conservative. It identifies clearly invalid
tracker-to-storm associations and limited temporal coverage without
classifying large forecast errors themselves as tracker failures.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .comparison import TrackVerification


@dataclass(slots=True)
class TrackerQCResult:
    """
    Quality-control result for one tracker verification.

    Parameters
    ----------
    status
        One of ``"pass"``, ``"limited"``, or ``"fail"``.

    reason
        Machine-readable QC reason.

    overlap_count
        Number of forecast/observation matches.

    initial_separation_km
        Track error at the first matched valid time.

    valid_for_aggregation
        Whether the result should be included in standard aggregate
        verification statistics.
    """

    status: str
    reason: str | None
    overlap_count: int
    initial_separation_km: float | None
    valid_for_aggregation: bool


def evaluate_tracker_qc(
    verification: TrackVerification,
    *,
    maximum_initial_separation_km: float = 2000.0,
    minimum_overlap: int = 6,
) -> TrackerQCResult:
    """
    Evaluate tracker-to-storm association quality.

    Notes
    -----
    This function does not classify a forecast as invalid merely
    because later track errors become large. Large forecast errors
    may represent genuine model error rather than tracker failure.
    """

    if not isinstance(
        verification,
        TrackVerification,
    ):
        raise TypeError(
            "verification must be a TrackVerification."
        )

    if maximum_initial_separation_km <= 0.0:
        raise ValueError(
            "maximum_initial_separation_km must be positive."
        )

    if minimum_overlap < 1:
        raise ValueError(
            "minimum_overlap must be at least 1."
        )

    overlap_count = (
        verification.overlap_count
    )

    if overlap_count == 0:
        return TrackerQCResult(
            status="fail",
            reason="no_overlap",
            overlap_count=0,
            initial_separation_km=None,
            valid_for_aggregation=False,
        )

    table = verification.table

    if "track_error_km" not in table.columns:
        raise ValueError(
            "verification table must contain "
            "'track_error_km'."
        )

    initial_error = float(
        table["track_error_km"].iloc[0]
    )

    if not np.isfinite(
        initial_error
    ):
        return TrackerQCResult(
            status="fail",
            reason="invalid_initial_separation",
            overlap_count=overlap_count,
            initial_separation_km=None,
            valid_for_aggregation=False,
        )

    if (
        initial_error
        > maximum_initial_separation_km
    ):
        return TrackerQCResult(
            status="fail",
            reason="initial_association",
            overlap_count=overlap_count,
            initial_separation_km=initial_error,
            valid_for_aggregation=False,
        )

    if overlap_count < minimum_overlap:
        return TrackerQCResult(
            status="limited",
            reason="short_overlap",
            overlap_count=overlap_count,
            initial_separation_km=initial_error,
            valid_for_aggregation=False,
        )

    return TrackerQCResult(
        status="pass",
        reason=None,
        overlap_count=overlap_count,
        initial_separation_km=initial_error,
        valid_for_aggregation=True,
    )