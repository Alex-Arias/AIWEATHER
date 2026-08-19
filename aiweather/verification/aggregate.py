"""
Aggregate tropical cyclone verification statistics.

This module provides utilities for summarizing multi-case
tropical cyclone verification results after tracker quality
control has been applied.

Aggregation is performed from case-level verification metrics.
Each valid forecast case therefore contributes equally to the
aggregate statistics, regardless of the number of verification
points available for that case.
"""

from __future__ import annotations

import pandas as pd


_REQUIRED_COLUMNS = {
    "model_name",
    "tracker",
    "coverage",
    "overlap_count",
    "mean_track_error_km",
    "rmse_track_error_km",
    "pressure_mae_pa",
    "wind_mae_ms",
    "valid_for_aggregation",
}


def aggregate_verification_summary(
    summary: pd.DataFrame,
) -> pd.DataFrame:
    """
    Aggregate valid multi-case verification statistics.

    Parameters
    ----------
    summary
        Batch verification summary produced by
        ``run_tc_verification_batch``.

        Only rows where ``valid_for_aggregation`` is True
        are included.

    Returns
    -------
    pandas.DataFrame
        Aggregate statistics grouped independently by model,
        tracker, and coverage.

    Notes
    -----
    Statistics are calculated from case-level metrics rather
    than by pooling individual verification points. This gives
    each valid forecast case equal weight in the case-level
    mean and median statistics.

    ``total_overlap_points`` is retained as a diagnostic of
    the total amount of verification data represented by each
    aggregate group.
    """

    if not isinstance(
        summary,
        pd.DataFrame,
    ):
        raise TypeError(
            "summary must be a pandas DataFrame."
        )

    missing = (
        _REQUIRED_COLUMNS
        - set(summary.columns)
    )

    if missing:
        missing_text = ", ".join(
            sorted(missing)
        )

        raise ValueError(
            "summary is missing required columns: "
            f"{missing_text}"
        )

    valid = summary.loc[
        summary[
            "valid_for_aggregation"
        ].eq(True)
    ].copy()

    columns = [
        "model_name",
        "tracker",
        "coverage",
        "case_count",
        "total_overlap_points",
        "mean_case_track_error_km",
        "median_case_track_error_km",
        "mean_case_rmse_track_error_km",
        "mean_pressure_mae_pa",
        "mean_wind_mae_ms",
    ]

    if valid.empty:
        return pd.DataFrame(
            columns=columns
        )

    aggregate = (
        valid.groupby(
            [
                "model_name",
                "tracker",
                "coverage",
            ],
            dropna=False,
            sort=True,
        )
        .agg(
            case_count=(
                "case_id",
                "nunique",
            ),
            total_overlap_points=(
                "overlap_count",
                "sum",
            ),
            mean_case_track_error_km=(
                "mean_track_error_km",
                "mean",
            ),
            median_case_track_error_km=(
                "mean_track_error_km",
                "median",
            ),
            mean_case_rmse_track_error_km=(
                "rmse_track_error_km",
                "mean",
            ),
            mean_pressure_mae_pa=(
                "pressure_mae_pa",
                "mean",
            ),
            mean_wind_mae_ms=(
                "wind_mae_ms",
                "mean",
            ),
        )
        .reset_index()
    )

    return aggregate[
        columns
    ]