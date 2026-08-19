"""
Lead-time-binned tropical cyclone verification statistics.

This module summarizes verified tropical cyclone forecast errors
as a function of forecast lead time.

Only tracker/case combinations that have passed batch-level
quality control should be supplied to this module.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


DEFAULT_LEAD_TIME_BINS = (
    (0.0, 24.0),
    (24.0, 48.0),
    (48.0, 72.0),
    (72.0, 120.0),
    (120.0, 168.0),
    (168.0, 240.0),
)


@dataclass(frozen=True, slots=True)
class LeadTimeBin:
    """
    Forecast lead-time interval.

    Parameters
    ----------
    start_hours
        Inclusive lower bound.

    end_hours
        Upper bound.

        The upper bound is exclusive except for the final
        lead-time bin, whose endpoint is included.
    """

    start_hours: float
    end_hours: float


def _validate_bins(
    bins: tuple[
        tuple[float, float],
        ...
    ],
) -> tuple[LeadTimeBin, ...]:
    """
    Validate and normalize lead-time bins.
    """

    if not isinstance(
        bins,
        tuple,
    ):
        raise TypeError(
            "bins must be a tuple."
        )

    if not bins:
        raise ValueError(
            "bins cannot be empty."
        )

    normalized: list[
        LeadTimeBin
    ] = []

    previous_end = None

    for item in bins:

        if (
            not isinstance(item, tuple)
            or len(item) != 2
        ):
            raise TypeError(
                "each lead-time bin must be "
                "a two-element tuple."
            )

        start = float(
            item[0]
        )
        end = float(
            item[1]
        )

        if not np.isfinite(start):
            raise ValueError(
                "lead-time bin start must be finite."
            )

        if not np.isfinite(end):
            raise ValueError(
                "lead-time bin end must be finite."
            )

        if start < 0.0:
            raise ValueError(
                "lead-time bin start cannot be negative."
            )

        if start >= end:
            raise ValueError(
                "lead-time bin start must be less "
                "than lead-time bin end."
            )

        if (
            previous_end is not None
            and start < previous_end
        ):
            raise ValueError(
                "lead-time bins cannot overlap."
            )

        normalized.append(
            LeadTimeBin(
                start_hours=start,
                end_hours=end,
            )
        )

        previous_end = end

    return tuple(
        normalized
    )


def assign_lead_time_bins(
    lead_times,
    *,
    bins: tuple[
        tuple[float, float],
        ...
    ] = DEFAULT_LEAD_TIME_BINS,
) -> pd.Categorical:
    """
    Assign forecast lead times to non-overlapping bins.

    The lower bound of each bin is inclusive. The upper
    bound is exclusive, except for the final bin, whose
    endpoint is included.

    Examples
    --------
    With the default bins:

    ``24 h`` belongs to the ``24-48 h`` bin.

    ``48 h`` belongs to the ``48-72 h`` bin.

    ``240 h`` belongs to the final ``168-240 h`` bin.
    """

    normalized = _validate_bins(
        bins
    )

    values = pd.to_numeric(
        pd.Series(lead_times),
        errors="coerce",
    )

    labels = [
        (
            f"{item.start_hours:g}-"
            f"{item.end_hours:g}h"
        )
        for item in normalized
    ]

    assignments = pd.Series(
        pd.NA,
        index=values.index,
        dtype="object",
    )

    for index, item in enumerate(
        normalized
    ):

        if index == len(
            normalized
        ) - 1:
            mask = (
                values.ge(
                    item.start_hours
                )
                & values.le(
                    item.end_hours
                )
            )
        else:
            mask = (
                values.ge(
                    item.start_hours
                )
                & values.lt(
                    item.end_hours
                )
            )

        assignments.loc[
            mask
        ] = labels[index]

    return pd.Categorical(
        assignments,
        categories=labels,
        ordered=True,
    )


def summarize_lead_time_verification(
    table: pd.DataFrame,
    *,
    bins: tuple[
        tuple[float, float],
        ...
    ] = DEFAULT_LEAD_TIME_BINS,
) -> pd.DataFrame:
    """
    Summarize one verified tracker table by forecast lead time.

    Parameters
    ----------
    table
        Verification table containing ``lead_time_hours``,
        ``track_error_km``, ``pressure_error_pa``, and
        ``wind_error_ms``.

    bins
        Forecast lead-time bins.

    Returns
    -------
    pandas.DataFrame
        One row per populated lead-time bin.
    """

    if not isinstance(
        table,
        pd.DataFrame,
    ):
        raise TypeError(
            "table must be a pandas DataFrame."
        )

    required = {
        "lead_time_hours",
        "track_error_km",
        "pressure_error_pa",
        "wind_error_ms",
    }

    missing = (
        required
        - set(table.columns)
    )

    if missing:
        raise ValueError(
            "table is missing required columns: "
            + ", ".join(
                sorted(missing)
            )
        )

    normalized = _validate_bins(
        bins
    )

    data = table.copy()

    data[
        "lead_time_bin"
    ] = assign_lead_time_bins(
        data[
            "lead_time_hours"
        ],
        bins=bins,
    )

    data = data[
        data[
            "lead_time_bin"
        ].notna()
    ].copy()

    columns = [
        "lead_time_bin",
        "lead_time_start_hours",
        "lead_time_end_hours",
        "point_count",
        "mean_track_error_km",
        "rmse_track_error_km",
        "median_track_error_km",
        "pressure_mae_pa",
        "pressure_rmse_pa",
        "wind_mae_ms",
        "wind_rmse_ms",
    ]

    if data.empty:
        return pd.DataFrame(
            columns=columns
        )

    lookup = {
        (
            f"{item.start_hours:g}-"
            f"{item.end_hours:g}h"
        ): item
        for item in normalized
    }

    rows = []

    for (
        lead_time_bin,
        group,
    ) in data.groupby(
        "lead_time_bin",
        observed=True,
        sort=True,
    ):

        track_error = pd.to_numeric(
            group[
                "track_error_km"
            ],
            errors="coerce",
        )

        pressure_error = pd.to_numeric(
            group[
                "pressure_error_pa"
            ],
            errors="coerce",
        )

        wind_error = pd.to_numeric(
            group[
                "wind_error_ms"
            ],
            errors="coerce",
        )

        interval = lookup[
            str(lead_time_bin)
        ]

        rows.append(
            {
                "lead_time_bin":
                    str(
                        lead_time_bin
                    ),

                "lead_time_start_hours":
                    interval.start_hours,

                "lead_time_end_hours":
                    interval.end_hours,

                "point_count":
                    int(
                        len(group)
                    ),

                "mean_track_error_km":
                    track_error.mean(),

                "rmse_track_error_km":
                    np.sqrt(
                        np.nanmean(
                            np.square(
                                track_error
                            )
                        )
                    ),

                "median_track_error_km":
                    track_error.median(),

                "pressure_mae_pa":
                    pressure_error
                    .abs()
                    .mean(),

                "pressure_rmse_pa":
                    np.sqrt(
                        np.nanmean(
                            np.square(
                                pressure_error
                            )
                        )
                    ),

                "wind_mae_ms":
                    wind_error
                    .abs()
                    .mean(),

                "wind_rmse_ms":
                    np.sqrt(
                        np.nanmean(
                            np.square(
                                wind_error
                            )
                        )
                    ),
            }
        )

    return pd.DataFrame(
        rows,
        columns=columns,
    )

def aggregate_batch_lead_time_verification(
    cases,
    results,
    summary: pd.DataFrame,
    *,
    bins: tuple[
        tuple[float, float],
        ...
    ] = DEFAULT_LEAD_TIME_BINS,
) -> pd.DataFrame:
    """
    Aggregate QC-valid verification points by forecast lead time.

    Parameters
    ----------
    cases
        Batch verification cases.

    results
        Single-case verification pipeline results corresponding
        positionally to ``cases``.

    summary
        QC-aware batch summary produced by
        ``run_tc_verification_batch``.

    bins
        Forecast lead-time bins.

    Returns
    -------
    pandas.DataFrame
        Lead-time-binned verification statistics grouped by
        model and tracker.

    Notes
    -----
    Only full-coverage tracker/case rows marked
    ``valid_for_aggregation=True`` are included.

    Verification points are pooled within each lead-time bin.
    ``case_count`` reports the number of distinct forecast
    cases contributing to each bin.
    """

    if not isinstance(
        summary,
        pd.DataFrame,
    ):
        raise TypeError(
            "summary must be a pandas DataFrame."
        )

    if len(cases) != len(results):
        raise ValueError(
            "cases and results must have the same length."
        )

    required = {
        "forecast_path",
        "model_name",
        "tracker",
        "coverage",
        "valid_for_aggregation",
    }

    missing = (
        required
        - set(summary.columns)
    )

    if missing:
        raise ValueError(
            "summary is missing required columns: "
            + ", ".join(
                sorted(missing)
            )
        )

    normalized = _validate_bins(
        bins
    )

    valid_summary = summary[
        summary["coverage"].eq("full")
        & summary[
            "valid_for_aggregation"
        ].eq(True)
    ].copy()

    columns = [
        "model_name",
        "tracker",
        "lead_time_bin",
        "lead_time_start_hours",
        "lead_time_end_hours",
        "case_count",
        "point_count",
        "mean_track_error_km",
        "rmse_track_error_km",
        "median_track_error_km",
        "pressure_mae_pa",
        "pressure_rmse_pa",
        "wind_mae_ms",
        "wind_rmse_ms",
    ]

    if valid_summary.empty:
        return pd.DataFrame(
            columns=columns
        )

    records: list[
        pd.DataFrame
    ] = []

    for (
        case_index,
        (
            case,
            result,
        ),
    ) in enumerate(
        zip(
            cases,
            results,
            strict=True,
        )
    ):

        forecast_path = str(
            case.forecast_path
        )

        case_summary = valid_summary[
            valid_summary[
                "forecast_path"
            ].eq(
                forecast_path
            )
        ]

        if case_summary.empty:
            continue

        verifications = (
            result
            .verification
            .verifications
        )

        for (
            tracker_name,
            tracker_verification,
        ) in verifications.items():

            tracker_summary = case_summary[
                case_summary[
                    "tracker"
                ].eq(
                    tracker_name
                )
            ]

            if tracker_summary.empty:
                continue

            model_name = str(
                tracker_summary[
                    "model_name"
                ].iloc[0]
            )

            table = (
                tracker_verification
                .table
                .copy()
            )

            required_table = {
                "lead_time_hours",
                "track_error_km",
                "pressure_error_pa",
                "wind_error_ms",
            }

            missing_table = (
                required_table
                - set(table.columns)
            )

            if missing_table:
                raise ValueError(
                    "verification table is missing "
                    "required columns: "
                    + ", ".join(
                        sorted(
                            missing_table
                        )
                    )
                )

            table[
                "lead_time_bin"
            ] = assign_lead_time_bins(
                table[
                    "lead_time_hours"
                ],
                bins=bins,
            )

            table = table[
                table[
                    "lead_time_bin"
                ].notna()
            ].copy()

            if table.empty:
                continue

            table[
                "model_name"
            ] = model_name

            table[
                "tracker"
            ] = tracker_name

            table[
                "_case_index"
            ] = case_index

            records.append(
                table[
                    [
                        "model_name",
                        "tracker",
                        "_case_index",
                        "lead_time_bin",
                        "track_error_km",
                        "pressure_error_pa",
                        "wind_error_ms",
                    ]
                ]
            )

    if not records:
        return pd.DataFrame(
            columns=columns
        )

    data = pd.concat(
        records,
        ignore_index=True,
    )

    lookup = {
        (
            f"{item.start_hours:g}-"
            f"{item.end_hours:g}h"
        ): item
        for item in normalized
    }

    rows = []

    grouped = data.groupby(
        [
            "model_name",
            "tracker",
            "lead_time_bin",
        ],
        observed=True,
        sort=True,
    )

    for (
        model_name,
        tracker_name,
        lead_time_bin,
    ), group in grouped:

        track_error = pd.to_numeric(
            group[
                "track_error_km"
            ],
            errors="coerce",
        )

        pressure_error = pd.to_numeric(
            group[
                "pressure_error_pa"
            ],
            errors="coerce",
        )

        wind_error = pd.to_numeric(
            group[
                "wind_error_ms"
            ],
            errors="coerce",
        )

        interval = lookup[
            str(
                lead_time_bin
            )
        ]

        rows.append(
            {
                "model_name":
                    model_name,

                "tracker":
                    tracker_name,

                "lead_time_bin":
                    str(
                        lead_time_bin
                    ),

                "lead_time_start_hours":
                    interval.start_hours,

                "lead_time_end_hours":
                    interval.end_hours,

                "case_count":
                    int(
                        group[
                            "_case_index"
                        ].nunique()
                    ),

                "point_count":
                    int(
                        len(group)
                    ),

                "mean_track_error_km":
                    track_error.mean(),

                "rmse_track_error_km":
                    np.sqrt(
                        np.nanmean(
                            np.square(
                                track_error
                            )
                        )
                    ),

                "median_track_error_km":
                    track_error.median(),

                "pressure_mae_pa":
                    pressure_error
                    .abs()
                    .mean(),

                "pressure_rmse_pa":
                    np.sqrt(
                        np.nanmean(
                            np.square(
                                pressure_error
                            )
                        )
                    ),

                "wind_mae_ms":
                    wind_error
                    .abs()
                    .mean(),

                "wind_rmse_ms":
                    np.sqrt(
                        np.nanmean(
                            np.square(
                                wind_error
                            )
                        )
                    ),
            }
        )

    return pd.DataFrame(
        rows,
        columns=columns,
    )