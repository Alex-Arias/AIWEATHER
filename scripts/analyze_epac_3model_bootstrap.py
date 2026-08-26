#!/usr/bin/env python3
"""
Paired bootstrap uncertainty analysis for the EPAC three-model
tropical-cyclone benchmark.

Models
------
GraphCast
AIFS2
Pangu3

Input
-----
Strict three-model exact-common-time table:

    results/verification/comparison/
    epac_3model_5storm/
    epac_3model_common_points.csv

The analysis is performed on the robust sample:
storms with at least MINIMUM_STORM_COMMON_POINTS exact common times.

Important
---------
Forecast points from the same storm are temporally dependent.
Therefore this script reports point-level paired bootstrap confidence
intervals as exploratory uncertainty estimates, not as independent
storm-level population inference.

Outputs
-------
1. Overall paired bootstrap summary
2. Lead-time-bin paired bootstrap summary
3. Confidence-interval figure for track error
4. Confidence-interval figure for pressure absolute error
5. Confidence-interval figure for wind absolute error
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

OVERALL_OUTPUT = (
    OUTPUT_DIR
    / "epac_3model_bootstrap_overall.csv"
)

LEADTIME_OUTPUT = (
    OUTPUT_DIR
    / "epac_3model_bootstrap_leadtime.csv"
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

PAIRS = [
    (
        "graphcast",
        "aifs2",
    ),
    (
        "graphcast",
        "pangu3",
    ),
    (
        "aifs2",
        "pangu3",
    ),
]

LEAD_BINS = [
    {
        "name": "0-48 h",
        "minimum": 0,
        "maximum": 48,
    },
    {
        "name": "54-96 h",
        "minimum": 54,
        "maximum": 96,
    },
    {
        "name": "102-144 h",
        "minimum": 102,
        "maximum": 144,
    },
    {
        "name": "150-240 h",
        "minimum": 150,
        "maximum": 240,
    },
]

MINIMUM_STORM_COMMON_POINTS = 6

BOOTSTRAP_SAMPLES = 20000
CONFIDENCE_LEVEL = 0.95
RANDOM_SEED = 20260826


# ============================================================
# Helpers
# ============================================================

def select_bin(
    data: pd.DataFrame,
    minimum: int,
    maximum: int,
) -> pd.DataFrame:
    """
    Select one forecast lead-time interval.
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


def paired_bootstrap_mean_difference(
    left,
    right,
    *,
    samples=BOOTSTRAP_SAMPLES,
    confidence_level=CONFIDENCE_LEVEL,
    rng,
):
    """
    Bootstrap paired mean difference:

        left - right

    Positive:
        right model has smaller error.

    Negative:
        left model has smaller error.
    """

    left = np.asarray(
        left,
        dtype=float,
    )

    right = np.asarray(
        right,
        dtype=float,
    )

    valid = (
        np.isfinite(
            left
        )
        & np.isfinite(
            right
        )
    )

    left = left[
        valid
    ]

    right = right[
        valid
    ]

    if len(
        left
    ) == 0:
        return {
            "n":
                0,

            "mean_difference":
                np.nan,

            "ci_lower":
                np.nan,

            "ci_upper":
                np.nan,

            "probability_left_better":
                np.nan,

            "probability_right_better":
                np.nan,
        }

    difference = (
        left
        - right
    )

    n = len(
        difference
    )

    indices = rng.integers(
        0,
        n,
        size=(
            samples,
            n,
        ),
    )

    boot = difference[
        indices
    ].mean(
        axis=1
    )

    alpha = (
        1.0
        - confidence_level
    )

    lower = np.quantile(
        boot,
        alpha / 2.0,
    )

    upper = np.quantile(
        boot,
        1.0 - alpha / 2.0,
    )

    # If left-right < 0, left has smaller error.
    probability_left_better = float(
        np.mean(
            boot < 0.0
        )
    )

    probability_right_better = float(
        np.mean(
            boot > 0.0
        )
    )

    return {
        "n":
            n,

        "mean_difference":
            float(
                difference.mean()
            ),

        "ci_lower":
            float(
                lower
            ),

        "ci_upper":
            float(
                upper
            ),

        "probability_left_better":
            probability_left_better,

        "probability_right_better":
            probability_right_better,
    }


def build_pair_result(
    subset,
    *,
    model_left,
    model_right,
    metric,
    left_values,
    right_values,
    lead_bin,
    rng,
):
    """
    Build one pairwise bootstrap-result row.
    """

    result = paired_bootstrap_mean_difference(
        left_values,
        right_values,
        rng=rng,
    )

    storm_count = (
        subset[
            "case_id"
        ]
        .nunique()
    )

    return {
        "lead_bin":
            lead_bin,

        "model_left":
            model_left,

        "model_right":
            model_right,

        "pair":
            (
                f"{model_left}_vs_"
                f"{model_right}"
            ),

        "metric":
            metric,

        "n":
            result[
                "n"
            ],

        "storm_count":
            storm_count,

        "mean_difference":
            result[
                "mean_difference"
            ],

        "ci_lower":
            result[
                "ci_lower"
            ],

        "ci_upper":
            result[
                "ci_upper"
            ],

        "probability_left_better":
            result[
                "probability_left_better"
            ],

        "probability_right_better":
            result[
                "probability_right_better"
            ],

        "confidence_level":
            CONFIDENCE_LEVEL,

        "bootstrap_samples":
            BOOTSTRAP_SAMPLES,

        "single_storm_warning":
            (
                storm_count
                < 2
            ),
    }


def analyze_subset(
    subset,
    *,
    lead_bin,
    rng,
):
    """
    Analyze all model pairs and metrics in one subset.
    """

    rows = []

    for (
        model_left,
        model_right,
    ) in PAIRS:

        # ----------------------------------------------------
        # Track error
        # ----------------------------------------------------

        left_track = subset[
            f"track_error_km_{model_left}"
        ]

        right_track = subset[
            f"track_error_km_{model_right}"
        ]

        rows.append(
            build_pair_result(
                subset,
                model_left=model_left,
                model_right=model_right,
                metric="track_error_km",
                left_values=left_track,
                right_values=right_track,
                lead_bin=lead_bin,
                rng=rng,
            )
        )

        # ----------------------------------------------------
        # Pressure absolute error
        # ----------------------------------------------------

        left_pressure = (
            subset[
                f"pressure_error_pa_{model_left}"
            ]
            .abs()
            / 100.0
        )

        right_pressure = (
            subset[
                f"pressure_error_pa_{model_right}"
            ]
            .abs()
            / 100.0
        )

        rows.append(
            build_pair_result(
                subset,
                model_left=model_left,
                model_right=model_right,
                metric="pressure_absolute_error_hpa",
                left_values=left_pressure,
                right_values=right_pressure,
                lead_bin=lead_bin,
                rng=rng,
            )
        )

        # ----------------------------------------------------
        # Wind absolute error
        # ----------------------------------------------------

        left_wind = subset[
            f"wind_error_ms_{model_left}"
        ].abs()

        right_wind = subset[
            f"wind_error_ms_{model_right}"
        ].abs()

        rows.append(
            build_pair_result(
                subset,
                model_left=model_left,
                model_right=model_right,
                metric="wind_absolute_error_ms",
                left_values=left_wind,
                right_values=right_wind,
                lead_bin=lead_bin,
                rng=rng,
            )
        )

    return rows


def save_ci_figure(
    data,
    metric,
    ylabel,
    stem,
):
    """
    Plot mean paired differences with 95% CI.

    Negative:
        left model better.

    Positive:
        right model better.
    """

    subset = data[
        data[
            "metric"
        ]
        == metric
    ].copy()

    pair_order = [
        "graphcast_vs_aifs2",
        "graphcast_vs_pangu3",
        "aifs2_vs_pangu3",
    ]

    lead_order = [
        "0-48 h",
        "54-96 h",
        "102-144 h",
        "150-240 h",
    ]

    fig, axes = plt.subplots(
        1,
        4,
        figsize=(
            17,
            5,
        ),
        sharey=True,
    )

    for ax, lead_bin in zip(
        axes,
        lead_order,
    ):

        section = subset[
            subset[
                "lead_bin"
            ]
            == lead_bin
        ].copy()

        section[
            "pair"
        ] = pd.Categorical(
            section[
                "pair"
            ],
            categories=pair_order,
            ordered=True,
        )

        section = section.sort_values(
            "pair"
        )

        y = np.arange(
            len(
                section
            )
        )

        means = section[
            "mean_difference"
        ].to_numpy(
            dtype=float
        )

        lower = section[
            "ci_lower"
        ].to_numpy(
            dtype=float
        )

        upper = section[
            "ci_upper"
        ].to_numpy(
            dtype=float
        )

        xerr = np.vstack(
            [
                means - lower,
                upper - means,
            ]
        )

        ax.errorbar(
            means,
            y,
            xerr=xerr,
            fmt="o",
            capsize=4,
        )

        ax.axvline(
            0.0,
            linewidth=1.0,
        )

        labels = []

        for _, row in section.iterrows():

            left = MODEL_LABELS[
                row[
                    "model_left"
                ]
            ]

            right = MODEL_LABELS[
                row[
                    "model_right"
                ]
            ]

            labels.append(
                f"{left} - {right}"
            )

        ax.set_yticks(
            y
        )

        ax.set_yticklabels(
            labels
        )

        ax.set_title(
            lead_bin,
            fontweight="bold",
        )

        ax.set_xlabel(
            ylabel
        )

        ax.grid(
            True,
            axis="x",
            alpha=0.3,
        )

        if (
            not section.empty
            and section[
                "single_storm_warning"
            ].all()
        ):

            ax.text(
                0.5,
                0.95,
                "Single storm",
                transform=ax.transAxes,
                ha="center",
                va="top",
                fontsize=9,
            )

    fig.suptitle(
        "EPAC Three-Model Paired Bootstrap Differences\n"
        "95% point-level bootstrap confidence intervals",
        fontsize=14,
        fontweight="bold",
    )

    fig.tight_layout(
        rect=[
            0,
            0,
            1,
            0.92,
        ]
    )

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
# Read data
# ============================================================

data = pd.read_csv(
    INPUT_PATH,
    parse_dates=[
        "valid_time",
    ],
)


# ============================================================
# Determine robust base sample
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


print("=" * 96)
print(
    "EPAC THREE-MODEL PAIRED BOOTSTRAP ANALYSIS"
)
print("=" * 96)

print()
print(
    "Input:",
    INPUT_PATH,
)

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

print(
    "Bootstrap samples:",
    BOOTSTRAP_SAMPLES,
)

print(
    "Confidence level:",
    CONFIDENCE_LEVEL,
)

print()
print(
    "NOTE: point-level bootstrap CIs are exploratory because "
    "forecast points within storms are temporally dependent."
)


# ============================================================
# Random-number generator
# ============================================================

rng = np.random.default_rng(
    RANDOM_SEED
)


# ============================================================
# Overall analysis
# ============================================================

overall_rows = analyze_subset(
    robust,
    lead_bin="overall",
    rng=rng,
)

overall = pd.DataFrame(
    overall_rows
)

overall.to_csv(
    OVERALL_OUTPUT,
    index=False,
)


# ============================================================
# Lead-time analysis
# ============================================================

lead_rows = []

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

    lead_rows.extend(
        analyze_subset(
            subset,
            lead_bin=lead_bin[
                "name"
            ],
            rng=rng,
        )
    )


lead_summary = pd.DataFrame(
    lead_rows
)

lead_summary.to_csv(
    LEADTIME_OUTPUT,
    index=False,
)


# ============================================================
# Console output
# ============================================================

print()
print("=" * 96)
print(
    "OVERALL PAIRED BOOTSTRAP"
)
print("=" * 96)
print()

display = overall[
    [
        "pair",
        "metric",
        "n",
        "storm_count",
        "mean_difference",
        "ci_lower",
        "ci_upper",
        "probability_left_better",
        "probability_right_better",
    ]
].copy()

print(
    display.to_string(
        index=False,
        float_format=lambda x:
            f"{x:.3f}",
    )
)


print()
print("=" * 96)
print(
    "LEAD-TIME PAIRED BOOTSTRAP"
)
print("=" * 96)
print()

display = lead_summary[
    [
        "lead_bin",
        "pair",
        "metric",
        "n",
        "storm_count",
        "mean_difference",
        "ci_lower",
        "ci_upper",
        "probability_left_better",
        "probability_right_better",
        "single_storm_warning",
    ]
].copy()

print(
    display.to_string(
        index=False,
        float_format=lambda x:
            f"{x:.3f}",
    )
)


# ============================================================
# Figures
# ============================================================

track_png, track_pdf = (
    save_ci_figure(
        lead_summary,
        metric="track_error_km",
        ylabel="Mean paired track-error difference (km)",
        stem=(
            "epac_3model_bootstrap_"
            "track_difference"
        ),
    )
)

pressure_png, pressure_pdf = (
    save_ci_figure(
        lead_summary,
        metric=(
            "pressure_absolute_error_hpa"
        ),
        ylabel=(
            "Mean paired pressure |error| "
            "difference (hPa)"
        ),
        stem=(
            "epac_3model_bootstrap_"
            "pressure_difference"
        ),
    )
)

wind_png, wind_pdf = (
    save_ci_figure(
        lead_summary,
        metric=(
            "wind_absolute_error_ms"
        ),
        ylabel=(
            "Mean paired wind |error| "
            "difference (m/s)"
        ),
        stem=(
            "epac_3model_bootstrap_"
            "wind_difference"
        ),
    )
)


# ============================================================
# Output products
# ============================================================

print()
print("=" * 96)
print(
    "OUTPUT PRODUCTS"
)
print("=" * 96)
print()

print(
    "Overall bootstrap:",
    OVERALL_OUTPUT,
)

print(
    "Lead-time bootstrap:",
    LEADTIME_OUTPUT,
)

print(
    "Track CI figure:",
    track_png,
)

print(
    "Pressure CI figure:",
    pressure_png,
)

print(
    "Wind CI figure:",
    wind_png,
)