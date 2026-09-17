#!/usr/bin/env python3
"""
Three-model Eastern Pacific tropical-cyclone comparison.

Models
------
GraphCast
AIFS2
Pangu3

Primary comparison
------------------
WuDuan tracker, full tracker coverage, exact common valid times
across all three forecast models.

This script reads existing point-level batch verification products.
It does not rerun forecasts or tropical-cyclone tracking.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from aiweather.plotting.model_style import (
    model_color,
    model_label,
    model_marker,
)
import numpy as np
import pandas as pd


# ============================================================
# Configuration
# ============================================================

MINIMUM_COMMON_POINTS = 6
MODEL_PATHS = {
    "graphcast": Path(
        "results/verification/batch/"
        "graphcast_epac_5storm/"
        "batch_points.csv"
    ),
    "aifs2": Path(
        "results/verification/batch/"
        "aifs2_epac_5storm/"
        "batch_points.csv"
    ),
    "pangu3": Path(
        "results/verification/batch/"
        "pangu3_epac_5storm/"
        "batch_points.csv"
    ),
}

STORMS = [

    (
        "douglas_20260701T000000",
        "Douglas",
    ),
    (
        "elida_20260714T120000",
        "Elida",
    ),
    (
        "fausto_20260719T000000",
        "Fausto",
    ),
    (
        "genevieve_20260724T000000",
        "Genevieve",
    ),
    (
        "hernan_20260811T000000",
        "Hernan",
    ),
]

TRACKER = "wuduan"
COVERAGE = "full"

OUTPUT_DIR = Path(
    "results/verification/comparison/"
    "epac_3model_5storm"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

LEAD_TICKS = [
    0,
    24,
    48,
    72,
    96,
    120,
    144,
    168,
    192,
    216,
    240,
]


# ============================================================
# Helpers
# ============================================================

def to_lon180(values):
    """
    Convert longitude from 0-360 to -180 to 180.
    """

    values = np.asarray(
        values,
        dtype=float,
    )

    return (
        (values + 180.0) % 360.0
        - 180.0
    )


def load_model_points(
    path: Path,
    model: str,
) -> pd.DataFrame:
    """
    Load one model batch table and retain full WuDuan coverage.
    """

    if not path.exists():
        raise FileNotFoundError(
            f"Missing batch point table: {path}"
        )

    data = pd.read_csv(
        path,
        parse_dates=[
            "initialization_time",
            "valid_time",
        ],
    )

    required = {
        "case_id",
        "sid",
        "valid_time",
        "lead_time_hours",
        "tracker",
        "coverage",
        "forecast_latitude",
        "forecast_longitude",
        "observed_latitude",
        "observed_longitude",
        "track_error_km",
        "forecast_pressure_pa",
        "observed_pressure_pa",
        "pressure_error_pa",
        "forecast_wind_ms",
        "observed_wind_ms",
        "wind_error_ms",
    }

    missing = required.difference(
        data.columns
    )

    if missing:
        raise ValueError(
            f"{model}: missing required columns: "
            f"{sorted(missing)}"
        )

    data = data[
        (
            data["tracker"]
            == TRACKER
        )
        & (
            data["coverage"]
            == COVERAGE
        )
    ].copy()

    data[
        "forecast_longitude_plot"
    ] = to_lon180(
        data["forecast_longitude"]
    )

    data[
        "observed_longitude_plot"
    ] = to_lon180(
        data["observed_longitude"]
    )

    data[
        "comparison_model"
    ] = model

    return (
        data
        .sort_values(
            [
                "case_id",
                "valid_time",
            ]
        )
        .reset_index(
            drop=True
        )
    )


def model_comparison_table(
    data: pd.DataFrame,
    model: str,
) -> pd.DataFrame:
    """
    Prepare one model for exact-time merging.
    """

    columns = [
        "case_id",
        "sid",
        "valid_time",
        "lead_time_hours",
        "forecast_latitude",
        "forecast_longitude",
        "forecast_longitude_plot",
        "observed_latitude",
        "observed_longitude",
        "observed_longitude_plot",
        "track_error_km",
        "forecast_pressure_pa",
        "observed_pressure_pa",
        "pressure_error_pa",
        "forecast_wind_ms",
        "observed_wind_ms",
        "wind_error_ms",
    ]

    table = data[
        columns
    ].copy()

    rename = {}

    for column in columns:

        if column in [
            "case_id",
            "sid",
            "valid_time",
        ]:
            continue

        rename[
            column
        ] = (
            f"{column}_{model}"
        )

    return table.rename(
        columns=rename
    )


def rmse(values):
    """
    Root-mean-square value.
    """

    values = np.asarray(
        values,
        dtype=float,
    )

    return float(
        np.sqrt(
            np.nanmean(
                values ** 2
            )
        )
    )


def save_figure(
    fig,
    stem: str,
):
    """
    Save PNG and PDF versions of one figure.
    """

    png = (
        OUTPUT_DIR
        / f"{stem}.png"
    )

    pdf = (
        OUTPUT_DIR
        / f"{stem}.pdf"
    )

    fig.savefig(
        png,
        dpi=300,
        bbox_inches="tight",
    )

    fig.savefig(
        pdf,
        bbox_inches="tight",
    )

    plt.close(
        fig
    )

    return (
        png,
        pdf,
    )


# ============================================================
# Load models
# ============================================================

model_data = {}

for model, path in MODEL_PATHS.items():

    model_data[
        model
    ] = load_model_points(
        path,
        model,
    )

    print(
        f"{model_label(model):<10s}: "
        f"{len(model_data[model])} "
        "full WuDuan points"
    )


# ============================================================
# Exact three-model common-time merge
# ============================================================

graphcast = model_comparison_table(
    model_data["graphcast"],
    "graphcast",
)

aifs2 = model_comparison_table(
    model_data["aifs2"],
    "aifs2",
)

pangu3 = model_comparison_table(
    model_data["pangu3"],
    "pangu3",
)


common = graphcast.merge(
    aifs2,
    on=[
        "case_id",
        "sid",
        "valid_time",
    ],
    how="inner",
    validate="one_to_one",
)

common = common.merge(
    pangu3,
    on=[
        "case_id",
        "sid",
        "valid_time",
    ],
    how="inner",
    validate="one_to_one",
)


# ============================================================
# Verify lead-time consistency
# ============================================================

lead_columns = [
    "lead_time_hours_graphcast",
    "lead_time_hours_aifs2",
    "lead_time_hours_pangu3",
]

lead_match = (
    (
        common[
            lead_columns[0]
        ]
        == common[
            lead_columns[1]
        ]
    )
    & (
        common[
            lead_columns[0]
        ]
        == common[
            lead_columns[2]
        ]
    )
)

common[
    "lead_times_match"
] = lead_match

if not common[
    "lead_times_match"
].all():

    bad = common[
        ~common[
            "lead_times_match"
        ]
    ]

    raise RuntimeError(
        "Common valid times contain inconsistent "
        "forecast lead times.\n"
        + bad[
            [
                "case_id",
                "valid_time",
                *lead_columns,
            ]
        ].to_string(
            index=False
        )
    )


common[
    "lead_time_hours"
] = common[
    "lead_time_hours_graphcast"
]


# ============================================================
# Verify common observed positions
# ============================================================

lat_columns = [
    "observed_latitude_graphcast",
    "observed_latitude_aifs2",
    "observed_latitude_pangu3",
]

lon_columns = [
    "observed_longitude_graphcast",
    "observed_longitude_aifs2",
    "observed_longitude_pangu3",
]

common[
    "observed_latitude"
] = common[
    lat_columns[0]
]

common[
    "observed_longitude"
] = common[
    lon_columns[0]
]

common[
    "observed_longitude_plot"
] = to_lon180(
    common[
        "observed_longitude"
    ]
)


# ============================================================
# Model wins at exact common times
# ============================================================

error_columns = {
    "graphcast":
        "track_error_km_graphcast",
    "aifs2":
        "track_error_km_aifs2",
    "pangu3":
        "track_error_km_pangu3",
}

error_matrix = common[
    list(
        error_columns.values()
    )
].to_numpy(
    dtype=float
)

winner_indices = np.nanargmin(
    error_matrix,
    axis=1,
)

winner_names = np.array(
    list(
        error_columns.keys()
    )
)

common[
    "best_model"
] = winner_names[
    winner_indices
]


# ============================================================
# Pairwise error differences
#
# Positive value means second model has lower error.
# ============================================================

common[
    "graphcast_minus_aifs2_km"
] = (
    common[
        "track_error_km_graphcast"
    ]
    - common[
        "track_error_km_aifs2"
    ]
)

common[
    "graphcast_minus_pangu3_km"
] = (
    common[
        "track_error_km_graphcast"
    ]
    - common[
        "track_error_km_pangu3"
    ]
)

common[
    "aifs2_minus_pangu3_km"
] = (
    common[
        "track_error_km_aifs2"
    ]
    - common[
        "track_error_km_pangu3"
    ]
)


# ============================================================
# Save common-point table
# ============================================================

common = common.sort_values(
    [
        "case_id",
        "valid_time",
    ]
).reset_index(
    drop=True
)

common_path = (
    OUTPUT_DIR
    / "epac_3model_common_points.csv"
)

common.to_csv(
    common_path,
    index=False,
)


# ============================================================
# Storm-level summary
# ============================================================

summary_rows = []


def append_summary(
    subset: pd.DataFrame,
    case_id: str,
    storm_name: str,
):

    if subset.empty:
        return

    row = {
        "case_id":
            case_id,

        "storm":
            storm_name,

        "common_points":
            len(subset),

        "first_lead_time_h":
            int(
                subset[
                    "lead_time_hours"
                ].min()
            ),

        "last_lead_time_h":
            int(
                subset[
                    "lead_time_hours"
                ].max()
            ),
    }

    for model in [
        "graphcast",
        "aifs2",
        "pangu3",
    ]:

        errors = subset[
            error_columns[
                model
            ]
        ]

        row[
            f"{model}_mean_error_km"
        ] = float(
            errors.mean()
        )

        row[
            f"{model}_rmse_km"
        ] = rmse(
            errors
        )

        row[
            f"{model}_median_error_km"
        ] = float(
            errors.median()
        )

        row[
            f"{model}_maximum_error_km"
        ] = float(
            errors.max()
        )

        row[
            f"{model}_best_points"
        ] = int(
            (
                subset[
                    "best_model"
                ]
                == model
            ).sum()
        )

        row[
            f"{model}_best_fraction"
        ] = float(
            (
                subset[
                    "best_model"
                ]
                == model
            ).mean()
        )

        pressure_error_hpa = (
            subset[
                f"pressure_error_pa_{model}"
            ]
            / 100.0
        )

        wind_error = subset[
            f"wind_error_ms_{model}"
        ]

        row[
            f"{model}_pressure_bias_hpa"
        ] = float(
            pressure_error_hpa.mean()
        )

        row[
            f"{model}_pressure_mae_hpa"
        ] = float(
            pressure_error_hpa.abs().mean()
        )

        row[
            f"{model}_pressure_rmse_hpa"
        ] = rmse(
            pressure_error_hpa
        )

        row[
            f"{model}_wind_bias_ms"
        ] = float(
            wind_error.mean()
        )

        row[
            f"{model}_wind_mae_ms"
        ] = float(
            wind_error.abs().mean()
        )

        row[
            f"{model}_wind_rmse_ms"
        ] = rmse(
            wind_error
        )

    row[
        "mean_graphcast_minus_aifs2_km"
    ] = float(
        subset[
            "graphcast_minus_aifs2_km"
        ].mean()
    )

    row[
        "mean_graphcast_minus_pangu3_km"
    ] = float(
        subset[
            "graphcast_minus_pangu3_km"
        ].mean()
    )

    row[
        "mean_aifs2_minus_pangu3_km"
    ] = float(
        subset[
            "aifs2_minus_pangu3_km"
        ].mean()
    )

    summary_rows.append(
        row
    )


for (
    case_id,
    storm_name,
) in STORMS:

    subset = common[
        common[
            "case_id"
        ]
        == case_id
    ].copy()

    append_summary(
        subset,
        case_id,
        storm_name,
    )


# ============================================================
# Pooled exact-common-time summary
# ============================================================


case_counts = (
    common.groupby(
        "case_id"
    )
    .size()
)

eligible_cases = (
    case_counts[
        case_counts
        >= MINIMUM_COMMON_POINTS
    ]
    .index
)

robust_common = common[
    common[
        "case_id"
    ].isin(
        eligible_cases
    )
].copy()

append_summary(
    robust_common,
    "robust_common",
    "Robust common sample",
)

summary = pd.DataFrame(
    summary_rows
)

summary_path = (
    OUTPUT_DIR
    / "epac_3model_summary.csv"
)

summary.to_csv(
    summary_path,
    index=False,
)


# ============================================================
# Coverage diagnostic
# ============================================================

coverage_rows = []

for (
    case_id,
    storm_name,
) in STORMS:

    row = {
        "case_id":
            case_id,

        "storm":
            storm_name,
    }

    for model in [
        "graphcast",
        "aifs2",
        "pangu3",
    ]:

        subset = model_data[
            model
        ]

        subset = subset[
            subset[
                "case_id"
            ]
            == case_id
        ]

        row[
            f"{model}_full_wuduan_points"
        ] = len(
            subset
        )

    common_subset = common[
        common[
            "case_id"
        ]
        == case_id
    ]

    row[
        "three_model_common_points"
    ] = len(
        common_subset
    )

    row[
        "eligible_for_comparison"
    ] = (
        len(
            common_subset
        )
        >= MINIMUM_COMMON_POINTS
    )

    coverage_rows.append(
        row
    )


coverage = pd.DataFrame(
    coverage_rows
)

coverage_path = (
    OUTPUT_DIR
    / "epac_3model_coverage.csv"
)

coverage.to_csv(
    coverage_path,
    index=False,
)


# ============================================================
# FIGURE 1
# Three-model tracks by storm
# ============================================================

fig, axes = plt.subplots(
    2,
    3,
    figsize=(
        18,
        10,
    ),
    sharey=True,
)

axes = np.asarray(
    axes
).ravel()

for ax, (
    case_id,
    storm_name,
) in zip(
    axes,
    STORMS,
):

    subset = common[
        common[
            "case_id"
        ]
        == case_id
    ].sort_values(
        "valid_time"
    )

    if subset.empty:

        ax.text(
            0.5,
            0.5,
            "No three-model\ncommon times",
            transform=ax.transAxes,
            ha="center",
            va="center",
        )

        continue

    ax.plot(
        subset[
            "observed_longitude_plot"
        ],
        subset[
            "observed_latitude"
        ],
        marker="o",
        markersize=4,
        linewidth=2.6,
        label="IBTrACS",
    )

    for model in [
        "graphcast",
        "aifs2",
        "pangu3",
    ]:

        ax.plot(
            to_lon180(
                subset[
                    f"forecast_longitude_{model}"
                ]
            ),
            subset[
                f"forecast_latitude_{model}"
            ],
            marker=model_marker(
                model
            ),
            markersize=3.5,
            linewidth=1.9,
            color=model_color(
                model
            ),
            label=model_label(
                model
            ),
        )

    ax.set_title(
        storm_name,
        fontweight="bold",
    )

    ax.set_xlabel(
        "Longitude (°)"
    )

    ax.grid(
        True,
        alpha=0.3,
    )

for ax in axes[
    len(STORMS):
]:
    ax.set_visible(
        False
    )

fig.supylabel(
    "Latitude (°)"
)

axes[0].legend(
    loc="best",
    fontsize=8,
)

fig.suptitle(
    "Eastern Pacific Tropical Cyclone Tracks\n"
    "IBTrACS vs GraphCast vs AIFS2 vs Pangu3 — "
    "WuDuan Exact Common Times",
    fontsize=14,
    fontweight="bold",
)

fig.tight_layout(
    rect=[
        0,
        0,
        1,
        0.91,
    ]
)

tracks_png, tracks_pdf = save_figure(
    fig,
    "epac_3model_tracks_by_storm",
)


# ============================================================
# FIGURE 2
# Track error versus lead time
# ============================================================

fig, axes = plt.subplots(
    2,
    3,
    figsize=(
        18,
        10,
    ),
    sharey=True,
)

axes = np.asarray(
    axes
).ravel()

for ax, (
    case_id,
    storm_name,
) in zip(
    axes,
    STORMS,
):

    subset = common[
        common[
            "case_id"
        ]
        == case_id
    ].sort_values(
        "lead_time_hours"
    )

    if subset.empty:

        ax.text(
            0.5,
            0.5,
            "No three-model\ncommon times",
            transform=ax.transAxes,
            ha="center",
            va="center",
        )

    else:

        for model in [
            "graphcast",
            "aifs2",
            "pangu3",
        ]:

            ax.plot(
                subset[
                    "lead_time_hours"
                ],
                subset[
                    error_columns[
                        model
                    ]
                ],
                marker=model_marker(
                    model
                ),
                markersize=4,
                linewidth=2,
                color=model_color(
                    model
                ),
                label=model_label(
                    model
                ),
            )

    ax.set_title(
        storm_name,
        fontweight="bold",
    )

    ax.set_xlabel(
        "Forecast lead time (h)"
    )

    ax.set_xlim(
        0,
        240,
    )

    ax.set_xticks(
        LEAD_TICKS
    )

    ax.tick_params(
        axis="x",
        rotation=45,
    )

    ax.grid(
        True,
        alpha=0.3,
    )

for ax in axes[
    len(STORMS):
]:
    ax.set_visible(
        False
    )

fig.supylabel(
    "Track error (km)"
)

axes[0].legend(
    loc="best",
    fontsize=8,
)

fig.suptitle(
    "Three-Model Tropical Cyclone Track Error\n"
    "WuDuan Tracker — Exact Common Valid Times",
    fontsize=14,
    fontweight="bold",
)

fig.tight_layout(
    rect=[
        0,
        0,
        1,
        0.91,
    ]
)

error_png, error_pdf = save_figure(
    fig,
    "epac_3model_track_error_by_storm",
)


# ============================================================
# FIGURE 3
# Storm-level mean track error
# ============================================================

storm_case_ids = [
    case_id
    for case_id, _ in STORMS
]

storm_summary = summary[
    summary[
        "case_id"
    ].isin(
        storm_case_ids
    )
].copy()

x = np.arange(
    len(
        storm_summary
    )
)

width = 0.24

fig, ax = plt.subplots(
    figsize=(
        11,
        6,
    )
)

for offset, model in zip(
    [
        -width,
        0.0,
        width,
    ],
    [
        "graphcast",
        "aifs2",
        "pangu3",
    ],
):

    ax.bar(
        x + offset,
        storm_summary[
            f"{model}_mean_error_km"
        ],
        width,
        color=model_color(
            model
        ),
        label=model_label(
            model
        ),
    )


ax.set_xticks(
    x
)

ax.set_xticklabels(
    storm_summary[
        "storm"
    ]
)

ax.set_ylabel(
    "Mean track error (km)"
)

ax.set_title(
    "Three-Model EPAC Track-Error Comparison\n"
    "WuDuan Tracker — Exact Common Valid Times"
)

ax.grid(
    True,
    axis="y",
    alpha=0.3,
)

ax.legend()

fig.tight_layout()

summary_png, summary_pdf = save_figure(
    fig,
    "epac_3model_mean_error_summary",
)


# ============================================================
# FIGURE 4
# Pressure error versus lead time
# ============================================================

fig, axes = plt.subplots(
    2,
    3,
    figsize=(
        18,
        10,
    ),
    sharey=True,
)

axes = np.asarray(
    axes
).ravel()

for ax, (
    case_id,
    storm_name,
) in zip(
    axes,
    STORMS,
):

    subset = common[
        common[
            "case_id"
        ]
        == case_id
    ].sort_values(
        "lead_time_hours"
    )

    if subset.empty:

        ax.text(
            0.5,
            0.5,
            "No three-model\ncommon times",
            transform=ax.transAxes,
            ha="center",
            va="center",
        )

    else:

        for model in [
            "graphcast",
            "aifs2",
            "pangu3",
        ]:

            pressure_error_hpa = (
                subset[
                    f"pressure_error_pa_{model}"
                ]
                / 100.0
            )

            ax.plot(
                subset[
                    "lead_time_hours"
                ],
                pressure_error_hpa,
                marker=model_marker(
                    model
                ),
                markersize=4,
                linewidth=2,
                color=model_color(
                    model
                ),
                label=model_label(
                    model
                ),
            )

    ax.axhline(
        0.0,
        linewidth=1.0,
    )

    ax.set_title(
        storm_name,
        fontweight="bold",
    )

    ax.set_xlabel(
        "Forecast lead time (h)"
    )

    ax.set_xlim(
        0,
        240,
    )

    ax.set_xticks(
        LEAD_TICKS
    )

    ax.tick_params(
        axis="x",
        rotation=45,
    )

    ax.grid(
        True,
        alpha=0.3,
    )

for ax in axes[
    len(STORMS):
]:
    ax.set_visible(
        False
    )

fig.supylabel(
    "Central-pressure error (hPa)"
)

axes[0].legend(
    loc="best",
    fontsize=8,
)

fig.suptitle(
    "Three-Model Tropical Cyclone Pressure Error\n"
    "Forecast minus IBTrACS — WuDuan Exact Common Times",
    fontsize=14,
    fontweight="bold",
)

fig.tight_layout(
    rect=[
        0,
        0,
        1,
        0.91,
    ]
)

pressure_png, pressure_pdf = save_figure(
    fig,
    "epac_3model_pressure_error_by_storm",
)


# ============================================================
# FIGURE 5
# Wind error versus lead time
# ============================================================

fig, axes = plt.subplots(
    2,
    3,
    figsize=(
        18,
        10,
    ),
    sharey=True,
)

axes = np.asarray(
    axes
).ravel()

for ax, (
    case_id,
    storm_name,
) in zip(
    axes,
    STORMS,
):

    subset = common[
        common[
            "case_id"
        ]
        == case_id
    ].sort_values(
        "lead_time_hours"
    )

    if subset.empty:

        ax.text(
            0.5,
            0.5,
            "No three-model\ncommon times",
            transform=ax.transAxes,
            ha="center",
            va="center",
        )

    else:

        for model in [
            "graphcast",
            "aifs2",
            "pangu3",
        ]:

            ax.plot(
                subset[
                    "lead_time_hours"
                ],
                subset[
                    f"wind_error_ms_{model}"
                ],
                marker=model_marker(
                    model
                ),
                markersize=4,
                linewidth=2,
                color=model_color(
                    model
                ),
                label=model_label(
                    model
                ),
            )

    ax.axhline(
        0.0,
        linewidth=1.0,
    )

    ax.set_title(
        storm_name,
        fontweight="bold",
    )

    ax.set_xlabel(
        "Forecast lead time (h)"
    )

    ax.set_xlim(
        0,
        240,
    )

    ax.set_xticks(
        LEAD_TICKS
    )

    ax.tick_params(
        axis="x",
        rotation=45,
    )

    ax.grid(
        True,
        alpha=0.3,
    )

for ax in axes[
    len(STORMS):
]:
    ax.set_visible(
        False
    )

fig.supylabel(
    "Maximum-wind error (m/s)"
)

axes[0].legend(
    loc="best",
    fontsize=8,
)

fig.suptitle(
    "Three-Model Tropical Cyclone Wind Error\n"
    "Forecast minus IBTrACS — WuDuan Exact Common Times",
    fontsize=14,
    fontweight="bold",
)

fig.tight_layout(
    rect=[
        0,
        0,
        1,
        0.91,
    ]
)

wind_png, wind_pdf = save_figure(
    fig,
    "epac_3model_wind_error_by_storm",
)


# ============================================================
# Console summary
# ============================================================

print()
print("=" * 92)
print(
    "EPAC THREE-MODEL TROPICAL-CYCLONE COMPARISON"
)
print("=" * 92)

print()
print(
    "Tracker       :",
    TRACKER,
)

print(
    "Coverage      :",
    COVERAGE,
)

print(
    "Comparison    : exact common valid times "
    "across GraphCast, AIFS2, and Pangu3"
)

print()

display_columns = [
    "storm",
    "common_points",
    "first_lead_time_h",
    "last_lead_time_h",
    "graphcast_mean_error_km",
    "aifs2_mean_error_km",
    "pangu3_mean_error_km",
    "graphcast_rmse_km",
    "aifs2_rmse_km",
    "pangu3_rmse_km",
    "graphcast_best_points",
    "aifs2_best_points",
    "pangu3_best_points",
]

print(
    summary[
        display_columns
    ].to_string(
        index=False,
        float_format=lambda x:
            f"{x:.1f}",
    )
)

print()
print("=" * 92)
print("ROBUST INTENSITY BENCHMARK")
print("=" * 92)
print()

robust_row = summary[
    summary[
        "case_id"
    ]
    == "robust_common"
].iloc[0]

intensity_rows = []

for model in [
    "graphcast",
    "aifs2",
    "pangu3",
]:

    intensity_rows.append(
        {
            "Model":
                model_label(
                    model
                ),

            "P Bias (hPa)":
                robust_row[
                    f"{model}_pressure_bias_hpa"
                ],

            "P MAE (hPa)":
                robust_row[
                    f"{model}_pressure_mae_hpa"
                ],

            "P RMSE (hPa)":
                robust_row[
                    f"{model}_pressure_rmse_hpa"
                ],

            "W Bias (m/s)":
                robust_row[
                    f"{model}_wind_bias_ms"
                ],

            "W MAE (m/s)":
                robust_row[
                    f"{model}_wind_mae_ms"
                ],

            "W RMSE (m/s)":
                robust_row[
                    f"{model}_wind_rmse_ms"
                ],
        }
    )

intensity_display = pd.DataFrame(
    intensity_rows
)

print(
    intensity_display.to_string(
        index=False,
        float_format=lambda x:
            f"{x:.2f}",
    )
)

print()
print(
    "Robust sample:",
    int(
        robust_row[
            "common_points"
        ]
    ),
    "exact common points from",
    len(
        eligible_cases
    ),
    "eligible storms.",
)

print()
print("=" * 92)
print("COVERAGE")
print("=" * 92)
print()

print(
    coverage.to_string(
        index=False,
    )
)

print()
print("=" * 92)
print("OUTPUT PRODUCTS")
print("=" * 92)

print()
print(
    "Common points:",
    common_path,
)

print(
    "Summary:",
    summary_path,
)

print(
    "Coverage:",
    coverage_path,
)

print(
    "Tracks:",
    tracks_png,
)

print(
    "Track errors:",
    error_png,
)

print(
    "Mean-error summary:",
    summary_png,
)

print(
    "Pressure errors:",
    pressure_png,
)

print(
    "Wind errors:",
    wind_png,
)

print(
    "Minimum common points per storm:",
    MINIMUM_COMMON_POINTS,
)

if __name__ == "__main__":
    pass
