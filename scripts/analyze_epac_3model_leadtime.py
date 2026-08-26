#!/usr/bin/env python3
"""
Lead-time-stratified EPAC three-model tropical-cyclone comparison.

Models
------
GraphCast
AIFS2
Pangu3

Tracker
-------
WuDuan

Input
-----
Strict three-model exact-common-time table produced by:

    scripts/compare_epac_3model_tc.py

Output
------
1. Lead-time summary CSV
2. Track-error figure
3. Pressure-error figure
4. Wind-error figure

The script does not rerun forecasts or TC tracking.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ============================================================
# Configuration
# ============================================================

INPUT_PATH = Path(
    "results/verification/comparison/"
    "epac_3model_5storm/"
    "epac_3model_common_points.csv"
)

OUTPUT_DIR = Path(
    "results/verification/comparison/"
    "epac_3model_5storm"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

OUTPUT_SUMMARY = (
    OUTPUT_DIR
    / "epac_3model_leadtime_summary.csv"
)

MODELS = [
    "graphcast",
    "aifs2",
    "pangu3",
]

MODEL_LABELS = {
    "graphcast": "GraphCast",
    "aifs2": "AIFS2",
    "pangu3": "Pangu3",
}

MARKERS = {
    "graphcast": "s",
    "aifs2": "^",
    "pangu3": "D",
}

# ------------------------------------------------------------
# Lead-time bins
#
# These boundaries preserve the 6-hour verification cadence
# used by the strict three-model common-time comparison.
# ------------------------------------------------------------

LEAD_BINS = [
    {
        "name": "0-48 h",
        "short_name": "0_48h",
        "minimum": 0,
        "maximum": 48,
    },
    {
        "name": "54-96 h",
        "short_name": "54_96h",
        "minimum": 54,
        "maximum": 96,
    },
    {
        "name": "102-144 h",
        "short_name": "102_144h",
        "minimum": 102,
        "maximum": 144,
    },
    {
        "name": "150-240 h",
        "short_name": "150_240h",
        "minimum": 150,
        "maximum": 240,
    },
]

MINIMUM_STORM_COMMON_POINTS = 6


# ============================================================
# Helpers
# ============================================================

def rmse(values):
    """
    Root-mean-square error.
    """

    values = np.asarray(
        values,
        dtype=float,
    )

    values = values[
        np.isfinite(
            values
        )
    ]

    if len(values) == 0:
        return np.nan

    return float(
        np.sqrt(
            np.mean(
                values ** 2
            )
        )
    )


def select_bin(
    data: pd.DataFrame,
    minimum: int,
    maximum: int,
) -> pd.DataFrame:
    """
    Select one lead-time interval.
    """

    return data[
        (
            data[
                "lead_time_hours"
            ]
            >= minimum
        )
        & (
            data[
                "lead_time_hours"
            ]
            <= maximum
        )
    ].copy()


def save_figure(
    fig,
    stem: str,
):
    """
    Save PNG and PDF.
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
        dpi=200,
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
# Read strict three-model exact-common table
# ============================================================

data = pd.read_csv(
    INPUT_PATH,
    parse_dates=[
        "valid_time",
    ],
)

print("=" * 92)
print(
    "EPAC THREE-MODEL LEAD-TIME ANALYSIS"
)
print("=" * 92)

print()
print(
    "Input:",
    INPUT_PATH,
)

print(
    "Rows:",
    len(
        data
    ),
)


# ============================================================
# Determine robust storms
# ============================================================

storm_counts = (
    data.groupby(
        "case_id"
    )
    .size()
)

eligible_cases = (
    storm_counts[
        storm_counts
        >= MINIMUM_STORM_COMMON_POINTS
    ]
    .index
)

robust = data[
    data[
        "case_id"
    ].isin(
        eligible_cases
    )
].copy()

print()
print(
    "Eligible storms:",
    list(
        eligible_cases
    ),
)

print(
    "Robust common points:",
    len(
        robust
    ),
)


# ============================================================
# Calculate lead-time metrics
# ============================================================

summary_rows = []

for lead_bin in LEAD_BINS:

    subset = select_bin(
        robust,
        lead_bin[
            "minimum"
        ],
        lead_bin[
            "maximum"
        ],
    )

    if subset.empty:
        continue

    # --------------------------------------------------------
    # Determine pointwise track winner
    # --------------------------------------------------------

    track_matrix = np.column_stack(
        [
            subset[
                f"track_error_km_{model}"
            ].to_numpy(
                dtype=float
            )
            for model in MODELS
        ]
    )

    winner_index = np.nanargmin(
        track_matrix,
        axis=1,
    )

    winners = np.asarray(
        MODELS
    )[
        winner_index
    ]

    for model in MODELS:

        track_error = subset[
            f"track_error_km_{model}"
        ].astype(
            float
        )

        pressure_error_hpa = (
            subset[
                f"pressure_error_pa_{model}"
            ].astype(
                float
            )
            / 100.0
        )

        wind_error = subset[
            f"wind_error_ms_{model}"
        ].astype(
            float
        )

        summary_rows.append(
            {
                "lead_bin":
                    lead_bin[
                        "name"
                    ],

                "lead_bin_min_h":
                    lead_bin[
                        "minimum"
                    ],

                "lead_bin_max_h":
                    lead_bin[
                        "maximum"
                    ],

                "model":
                    model,

                "model_label":
                    MODEL_LABELS[
                        model
                    ],

                "common_points":
                    len(
                        subset
                    ),

                "storm_count":
                    subset[
                        "case_id"
                    ].nunique(),

                # --------------------------------------------
                # Track
                # --------------------------------------------

                "track_mean_error_km":
                    float(
                        track_error.mean()
                    ),

                "track_median_error_km":
                    float(
                        track_error.median()
                    ),

                "track_rmse_km":
                    rmse(
                        track_error
                    ),

                "track_maximum_error_km":
                    float(
                        track_error.max()
                    ),

                "track_best_points":
                    int(
                        np.sum(
                            winners
                            == model
                        )
                    ),

                "track_best_fraction":
                    float(
                        np.mean(
                            winners
                            == model
                        )
                    ),

                # --------------------------------------------
                # Central pressure
                # --------------------------------------------

                "pressure_bias_hpa":
                    float(
                        pressure_error_hpa.mean()
                    ),

                "pressure_mae_hpa":
                    float(
                        pressure_error_hpa.abs().mean()
                    ),

                "pressure_rmse_hpa":
                    rmse(
                        pressure_error_hpa
                    ),

                # --------------------------------------------
                # Maximum wind
                # --------------------------------------------

                "wind_bias_ms":
                    float(
                        wind_error.mean()
                    ),

                "wind_mae_ms":
                    float(
                        wind_error.abs().mean()
                    ),

                "wind_rmse_ms":
                    rmse(
                        wind_error
                    ),
            }
        )


summary = pd.DataFrame(
    summary_rows
)

summary.to_csv(
    OUTPUT_SUMMARY,
    index=False,
)


# ============================================================
# Console summary
# ============================================================

print()
print("=" * 92)
print(
    "LEAD-TIME-STRATIFIED TRACK SKILL"
)
print("=" * 92)
print()

track_columns = [
    "lead_bin",
    "model_label",
    "common_points",
    "track_mean_error_km",
    "track_rmse_km",
    "track_best_points",
    "track_best_fraction",
]

print(
    summary[
        track_columns
    ].to_string(
        index=False,
        float_format=lambda value:
            f"{value:.2f}",
    )
)


print()
print("=" * 92)
print(
    "LEAD-TIME-STRATIFIED INTENSITY SKILL"
)
print("=" * 92)
print()

intensity_columns = [
    "lead_bin",
    "model_label",
    "pressure_bias_hpa",
    "pressure_mae_hpa",
    "pressure_rmse_hpa",
    "wind_bias_ms",
    "wind_mae_ms",
    "wind_rmse_ms",
]

print(
    summary[
        intensity_columns
    ].to_string(
        index=False,
        float_format=lambda value:
            f"{value:.2f}",
    )
)


# ============================================================
# Figure 1
# Track error by lead-time bin
# ============================================================

fig, ax = plt.subplots(
    figsize=(
        11,
        6,
    )
)

x = np.arange(
    len(
        LEAD_BINS
    )
)

width = 0.24

for offset, model in zip(
    [
        -width,
        0.0,
        width,
    ],
    MODELS,
):

    values = []

    for lead_bin in LEAD_BINS:

        row = summary[
            (
                summary[
                    "lead_bin"
                ]
                == lead_bin[
                    "name"
                ]
            )
            & (
                summary[
                    "model"
                ]
                == model
            )
        ]

        if row.empty:
            values.append(
                np.nan
            )

        else:
            values.append(
                row[
                    "track_mean_error_km"
                ].iloc[
                    0
                ]
            )

    ax.bar(
        x + offset,
        values,
        width,
        label=MODEL_LABELS[
            model
        ],
    )


ax.set_xticks(
    x
)

ax.set_xticklabels(
    [
        lead_bin[
            "name"
        ]
        for lead_bin in LEAD_BINS
    ]
)

ax.set_xlabel(
    "Forecast lead-time range"
)

ax.set_ylabel(
    "Mean track error (km)"
)

ax.set_title(
    "EPAC Three-Model Track Skill by Forecast Range\n"
    "WuDuan — Robust Exact Common-Time Sample"
)

ax.grid(
    True,
    axis="y",
    alpha=0.3,
)

ax.legend()

fig.tight_layout()

track_png, track_pdf = save_figure(
    fig,
    "epac_3model_leadtime_track",
)


# ============================================================
# Figure 2
# Pressure MAE by lead-time bin
# ============================================================

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
    MODELS,
):

    values = []

    for lead_bin in LEAD_BINS:

        row = summary[
            (
                summary[
                    "lead_bin"
                ]
                == lead_bin[
                    "name"
                ]
            )
            & (
                summary[
                    "model"
                ]
                == model
            )
        ]

        if row.empty:
            values.append(
                np.nan
            )

        else:
            values.append(
                row[
                    "pressure_mae_hpa"
                ].iloc[
                    0
                ]
            )

    ax.bar(
        x + offset,
        values,
        width,
        label=MODEL_LABELS[
            model
        ],
    )


ax.set_xticks(
    x
)

ax.set_xticklabels(
    [
        lead_bin[
            "name"
        ]
        for lead_bin in LEAD_BINS
    ]
)

ax.set_xlabel(
    "Forecast lead-time range"
)

ax.set_ylabel(
    "Central-pressure MAE (hPa)"
)

ax.set_title(
    "EPAC Three-Model Central-Pressure Skill by Forecast Range\n"
    "WuDuan — Robust Exact Common-Time Sample"
)

ax.grid(
    True,
    axis="y",
    alpha=0.3,
)

ax.legend()

fig.tight_layout()

pressure_png, pressure_pdf = save_figure(
    fig,
    "epac_3model_leadtime_pressure",
)


# ============================================================
# Figure 3
# Wind MAE by lead-time bin
# ============================================================

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
    MODELS,
):

    values = []

    for lead_bin in LEAD_BINS:

        row = summary[
            (
                summary[
                    "lead_bin"
                ]
                == lead_bin[
                    "name"
                ]
            )
            & (
                summary[
                    "model"
                ]
                == model
            )
        ]

        if row.empty:
            values.append(
                np.nan
            )

        else:
            values.append(
                row[
                    "wind_mae_ms"
                ].iloc[
                    0
                ]
            )

    ax.bar(
        x + offset,
        values,
        width,
        label=MODEL_LABELS[
            model
        ],
    )


ax.set_xticks(
    x
)

ax.set_xticklabels(
    [
        lead_bin[
            "name"
        ]
        for lead_bin in LEAD_BINS
    ]
)

ax.set_xlabel(
    "Forecast lead-time range"
)

ax.set_ylabel(
    "Maximum-wind MAE (m/s)"
)

ax.set_title(
    "EPAC Three-Model Maximum-Wind Skill by Forecast Range\n"
    "WuDuan — Robust Exact Common-Time Sample"
)

ax.grid(
    True,
    axis="y",
    alpha=0.3,
)

ax.legend()

fig.tight_layout()

wind_png, wind_pdf = save_figure(
    fig,
    "epac_3model_leadtime_wind",
)


# ============================================================
# Output summary
# ============================================================

print()
print("=" * 92)
print(
    "OUTPUT PRODUCTS"
)
print("=" * 92)
print()

print(
    "Lead-time summary:",
    OUTPUT_SUMMARY,
)

print(
    "Track figure:",
    track_png,
)

print(
    "Pressure figure:",
    pressure_png,
)

print(
    "Wind figure:",
    wind_png,
)