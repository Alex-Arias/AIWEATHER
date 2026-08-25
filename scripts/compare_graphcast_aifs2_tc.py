"""
GraphCast vs AIFS2 tropical-cyclone comparison.

Products
--------
Primary model comparison:
    WuDuan tracker, full coverage.

Tracker-sensitivity comparison:
    Native vs WuDuan for GraphCast and AIFS2.

The script reads point-level batch verification products and does
not rerun forecasts or cyclone tracking.
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

GRAPHCAST_POINTS = Path(
    "results/verification/batch/"
    "graphcast_epac_4storm/"
    "batch_points.csv"
)

AIFS2_POINTS = Path(
    "results/verification/batch/"
    "aifs2_epac_4storm/"
    "batch_points.csv"
)

OUTPUT_DIR = Path(
    "results/verification/comparison/"
    "graphcast_vs_aifs2_epac_4storm"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

STORMS = [
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

COVERAGE = "full"

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


def load_points(
    path: Path,
) -> pd.DataFrame:
    """
    Read one point-level batch table.

    Keep all trackers but restrict the analysis to full
    tracker coverage.
    """

    data = pd.read_csv(
        path,
        parse_dates=[
            "initialization_time",
            "valid_time",
        ],
    )

    data = data[
        data["coverage"]
        == COVERAGE
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

    return data


def tracker_subset(
    data: pd.DataFrame,
    tracker: str,
) -> pd.DataFrame:
    """
    Return one tracker from a point-level table.
    """

    return (
        data[
            data["tracker"]
            == tracker
        ]
        .copy()
    )


def storm_subset(
    data: pd.DataFrame,
    case_id: str,
) -> pd.DataFrame:
    """
    Return one storm sorted by valid time.
    """

    return (
        data[
            data["case_id"]
            == case_id
        ]
        .sort_values(
            "valid_time"
        )
        .copy()
    )


def build_observations(
    *tables,
) -> pd.DataFrame:
    """
    Construct the longest available matched IBTrACS segment
    from one or more verification tables.
    """

    pieces = []

    for table in tables:

        if table.empty:
            continue

        pieces.append(
            table[
                [
                    "valid_time",
                    "observed_latitude",
                    "observed_longitude_plot",
                ]
            ]
        )

    if not pieces:
        return pd.DataFrame(
            columns=[
                "valid_time",
                "observed_latitude",
                "observed_longitude_plot",
            ]
        )

    return (
        pd.concat(
            pieces,
            ignore_index=True,
        )
        .drop_duplicates(
            subset=[
                "valid_time",
            ]
        )
        .sort_values(
            "valid_time"
        )
    )


def add_lead_annotations(
    ax,
    data: pd.DataFrame,
    marker: str,
):
    """
    Mark and label 24-hour forecast positions.
    """

    if data.empty:
        return

    selected = data[
        data[
            "lead_time_hours"
        ] % 24 == 0
    ]

    ax.scatter(
        selected[
            "forecast_longitude_plot"
        ],
        selected[
            "forecast_latitude"
        ],
        marker=marker,
        s=35,
        zorder=4,
    )

    for _, row in selected.iterrows():

        ax.annotate(
            f"{int(row['lead_time_hours'])}",
            (
                row[
                    "forecast_longitude_plot"
                ],
                row[
                    "forecast_latitude"
                ],
            ),
            xytext=(
                4,
                4,
            ),
            textcoords="offset points",
            fontsize=7,
        )


def save_figure(
    fig,
    stem: str,
):
    """
    Save one figure as PNG and PDF.
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
# Read complete point tables
# ============================================================

graphcast_all = load_points(
    GRAPHCAST_POINTS
)

aifs2_all = load_points(
    AIFS2_POINTS
)


# ============================================================
# Tracker-specific tables
# ============================================================

graphcast_wuduan = tracker_subset(
    graphcast_all,
    "wuduan",
)

aifs2_wuduan = tracker_subset(
    aifs2_all,
    "wuduan",
)

graphcast_native = tracker_subset(
    graphcast_all,
    "native",
)

aifs2_native = tracker_subset(
    aifs2_all,
    "native",
)


# ============================================================
# Primary paired comparison
# WuDuan only
# ============================================================

pair_columns = [
    "case_id",
    "sid",
    "valid_time",
    "lead_time_hours",
    "observed_latitude",
    "observed_longitude",
    "track_error_km",
    "forecast_latitude",
    "forecast_longitude",
    "forecast_pressure_pa",
    "observed_pressure_pa",
    "pressure_error_pa",
    "forecast_wind_ms",
    "observed_wind_ms",
    "wind_error_ms",
]

gc_pair = graphcast_wuduan[
    pair_columns
].copy()

ai_pair = aifs2_wuduan[
    pair_columns
].copy()

paired = gc_pair.merge(
    ai_pair,
    on=[
        "case_id",
        "sid",
        "valid_time",
    ],
    how="inner",
    suffixes=(
        "_graphcast",
        "_aifs2",
    ),
)

paired[
    "delta_track_error_km"
] = (
    paired[
        "track_error_km_graphcast"
    ]
    - paired[
        "track_error_km_aifs2"
    ]
)

paired[
    "aifs2_better"
] = (
    paired[
        "delta_track_error_km"
    ]
    > 0.0
)

paired[
    "graphcast_better"
] = (
    paired[
        "delta_track_error_km"
    ]
    < 0.0
)

paired[
    "equal_error"
] = (
    paired[
        "delta_track_error_km"
    ]
    == 0.0
)


# ============================================================
# Save paired WuDuan table
# ============================================================

paired_path = (
    OUTPUT_DIR
    / "paired_track_error.csv"
)

paired.to_csv(
    paired_path,
    index=False,
)


# ============================================================
# Storm-level WuDuan paired summary
# ============================================================

summary_rows = []

for (
    case_id,
    storm_name,
) in STORMS:

    subset = paired[
        paired["case_id"]
        == case_id
    ].copy()

    if subset.empty:
        continue

    gc_error = subset[
        "track_error_km_graphcast"
    ]

    ai_error = subset[
        "track_error_km_aifs2"
    ]

    delta = subset[
        "delta_track_error_km"
    ]

    summary_rows.append(
        {
            "case_id":
                case_id,

            "storm":
                storm_name,

            "paired_points":
                len(subset),

            "graphcast_mean_error_km":
                gc_error.mean(),

            "aifs2_mean_error_km":
                ai_error.mean(),

            "graphcast_rmse_km":
                np.sqrt(
                    np.mean(
                        gc_error ** 2
                    )
                ),

            "aifs2_rmse_km":
                np.sqrt(
                    np.mean(
                        ai_error ** 2
                    )
                ),

            "mean_delta_error_km":
                delta.mean(),

            "median_delta_error_km":
                delta.median(),

            "aifs2_better_points":
                int(
                    subset[
                        "aifs2_better"
                    ].sum()
                ),

            "graphcast_better_points":
                int(
                    subset[
                        "graphcast_better"
                    ].sum()
                ),

            "equal_error_points":
                int(
                    subset[
                        "equal_error"
                    ].sum()
                ),

            "aifs2_better_fraction":
                subset[
                    "aifs2_better"
                ].mean(),
        }
    )

summary = pd.DataFrame(
    summary_rows
)

summary_path = (
    OUTPUT_DIR
    / "paired_track_error_summary.csv"
)

summary.to_csv(
    summary_path,
    index=False,
)


# ============================================================
# Tracker-sensitivity numerical summary
# ============================================================

tracker_summary_rows = []

datasets = [
    (
        "graphcast",
        "native",
        graphcast_native,
    ),
    (
        "graphcast",
        "wuduan",
        graphcast_wuduan,
    ),
    (
        "aifs2",
        "native",
        aifs2_native,
    ),
    (
        "aifs2",
        "wuduan",
        aifs2_wuduan,
    ),
]

for (
    model_name,
    tracker_name,
    dataframe,
) in datasets:

    for (
        case_id,
        storm_name,
    ) in STORMS:

        subset = storm_subset(
            dataframe,
            case_id,
        )

        if subset.empty:
            continue

        errors = subset[
            "track_error_km"
        ]

        tracker_summary_rows.append(
            {
                "case_id":
                    case_id,

                "storm":
                    storm_name,

                "model_name":
                    model_name,

                "tracker":
                    tracker_name,

                "point_count":
                    len(subset),

                "mean_track_error_km":
                    errors.mean(),

                "rmse_track_error_km":
                    np.sqrt(
                        np.mean(
                            errors ** 2
                        )
                    ),

                "median_track_error_km":
                    errors.median(),

                "maximum_track_error_km":
                    errors.max(),

                "first_lead_time_hours":
                    subset[
                        "lead_time_hours"
                    ].min(),

                "last_lead_time_hours":
                    subset[
                        "lead_time_hours"
                    ].max(),
            }
        )

tracker_summary = pd.DataFrame(
    tracker_summary_rows
)

tracker_summary_path = (
    OUTPUT_DIR
    / "tracker_sensitivity_summary.csv"
)

tracker_summary.to_csv(
    tracker_summary_path,
    index=False,
)


# ============================================================
# FIGURE 1
# Existing WuDuan tracks by storm
# ============================================================

fig, axes = plt.subplots(
    1,
    4,
    figsize=(20, 5.5),
    sharey=True,
)

for ax, (
    case_id,
    storm_name,
) in zip(
    axes,
    STORMS,
):

    gc = storm_subset(
        graphcast_wuduan,
        case_id,
    )

    ai = storm_subset(
        aifs2_wuduan,
        case_id,
    )

    obs = build_observations(
        gc,
        ai,
    )

    ax.plot(
        obs[
            "observed_longitude_plot"
        ],
        obs[
            "observed_latitude"
        ],
        marker="o",
        markersize=4,
        linewidth=2.3,
        label="IBTrACS",
    )

    ax.plot(
        gc[
            "forecast_longitude_plot"
        ],
        gc[
            "forecast_latitude"
        ],
        marker="s",
        markersize=4,
        linewidth=1.8,
        label="GraphCast",
    )

    ax.plot(
        ai[
            "forecast_longitude_plot"
        ],
        ai[
            "forecast_latitude"
        ],
        marker="^",
        markersize=4,
        linewidth=1.8,
        label="AIFS2",
    )

    add_lead_annotations(
        ax,
        gc,
        "s",
    )

    add_lead_annotations(
        ax,
        ai,
        "^",
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

axes[0].set_ylabel(
    "Latitude (°)"
)

axes[0].legend(
    loc="best",
)

fig.suptitle(
    "Eastern Pacific Tropical Cyclone Tracks\n"
    "IBTrACS vs GraphCast vs AIFS2 — WuDuan Tracker",
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

track_png, track_pdf = save_figure(
    fig,
    "graphcast_vs_aifs2_tracks_by_storm",
)


# ============================================================
# FIGURE 2
# Existing WuDuan track error by storm
# ============================================================

fig, axes = plt.subplots(
    1,
    4,
    figsize=(20, 5.5),
    sharey=True,
)

for ax, (
    case_id,
    storm_name,
) in zip(
    axes,
    STORMS,
):

    gc = storm_subset(
        graphcast_wuduan,
        case_id,
    )

    ai = storm_subset(
        aifs2_wuduan,
        case_id,
    )

    ax.plot(
        gc[
            "lead_time_hours"
        ],
        gc[
            "track_error_km"
        ],
        marker="o",
        markersize=4,
        linewidth=2,
        label="GraphCast",
    )

    ax.plot(
        ai[
            "lead_time_hours"
        ],
        ai[
            "track_error_km"
        ],
        marker="s",
        markersize=4,
        linewidth=2,
        label="AIFS2",
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

axes[0].set_ylabel(
    "Track error (km)"
)

axes[0].legend(
    loc="best",
)

fig.suptitle(
    "Tropical Cyclone Track Error by Lead Time\n"
    "GraphCast vs AIFS2 — WuDuan Tracker",
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
    "graphcast_vs_aifs2_track_error_by_storm",
)


# ============================================================
# FIGURE 3
# Existing paired WuDuan model-error difference
# ============================================================

fig, axes = plt.subplots(
    1,
    4,
    figsize=(20, 5.5),
    sharey=True,
)

for ax, (
    case_id,
    storm_name,
) in zip(
    axes,
    STORMS,
):

    subset = (
        paired[
            paired["case_id"]
            == case_id
        ]
        .sort_values(
            "lead_time_hours_graphcast"
        )
    )

    ax.axhline(
        0.0,
        linewidth=1.2,
    )

    if subset.empty:
        ax.text(
            0.5,
            0.5,
            "No paired WuDuan\ncomparison available",
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=10,
            fontstyle="italic",
            alpha=0.7,
        )
    else:
        ax.plot(
            subset[
                "lead_time_hours_graphcast"
            ],
            subset[
                "delta_track_error_km"
            ],
            marker="o",
            markersize=4,
            linewidth=2,
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

axes[0].set_ylabel(
    "GraphCast error − AIFS2 error (km)"
)

fig.suptitle(
    "Paired Tropical Cyclone Track-Error Difference\n"
    "Positive Values Indicate Lower AIFS2 Error",
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

delta_png, delta_pdf = save_figure(
    fig,
    "graphcast_minus_aifs2_track_error",
)


# ============================================================
# FIGURE 4
# Native tracks by storm
# ============================================================

fig, axes = plt.subplots(
    1,
    4,
    figsize=(20, 5.5),
    sharey=True,
)

for ax, (
    case_id,
    storm_name,
) in zip(
    axes,
    STORMS,
):

    gc = storm_subset(
        graphcast_native,
        case_id,
    )

    ai = storm_subset(
        aifs2_native,
        case_id,
    )

    obs = build_observations(
        gc,
        ai,
    )

    ax.plot(
        obs[
            "observed_longitude_plot"
        ],
        obs[
            "observed_latitude"
        ],
        marker="o",
        markersize=4,
        linewidth=2.3,
        label="IBTrACS",
    )

    ax.plot(
        gc[
            "forecast_longitude_plot"
        ],
        gc[
            "forecast_latitude"
        ],
        marker="s",
        markersize=4,
        linewidth=1.8,
        label="GraphCast Native",
    )

    ax.plot(
        ai[
            "forecast_longitude_plot"
        ],
        ai[
            "forecast_latitude"
        ],
        marker="^",
        markersize=4,
        linewidth=1.8,
        label="AIFS2 Native",
    )

    add_lead_annotations(
        ax,
        gc,
        "s",
    )

    add_lead_annotations(
        ax,
        ai,
        "^",
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

axes[0].set_ylabel(
    "Latitude (°)"
)

axes[0].legend(
    loc="best",
)

fig.suptitle(
    "Eastern Pacific Tropical Cyclone Tracks\n"
    "IBTrACS vs GraphCast vs AIFS2 — Native Tracker",
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

native_track_png, native_track_pdf = save_figure(
    fig,
    "graphcast_vs_aifs2_native_tracks_by_storm",
)

# ============================================================
# FIGURE 5
# All tracks by storm
#
# IBTrACS
# GraphCast Native
# GraphCast WuDuan
# AIFS2 Native
# AIFS2 WuDuan
# ============================================================

fig, axes = plt.subplots(
    1,
    4,
    figsize=(20, 5.5),
    sharey=True,
)

for ax, (
    case_id,
    storm_name,
) in zip(
    axes,
    STORMS,
):

    gc_native = storm_subset(
        graphcast_native,
        case_id,
    )

    gc_wuduan = storm_subset(
        graphcast_wuduan,
        case_id,
    )

    ai_native = storm_subset(
        aifs2_native,
        case_id,
    )

    ai_wuduan = storm_subset(
        aifs2_wuduan,
        case_id,
    )

    obs = build_observations(
        gc_native,
        gc_wuduan,
        ai_native,
        ai_wuduan,
    )

    # --------------------------------------------------------
    # IBTrACS
    # --------------------------------------------------------

    ax.plot(
        obs[
            "observed_longitude_plot"
        ],
        obs[
            "observed_latitude"
        ],
        linewidth=2.8,
        marker="o",
        markersize=4,
        label="IBTrACS",
        zorder=5,
    )

    # --------------------------------------------------------
    # GraphCast
    # --------------------------------------------------------

    ax.plot(
        gc_native[
            "forecast_longitude_plot"
        ],
        gc_native[
            "forecast_latitude"
        ],
        linestyle="--",
        linewidth=1.8,
        marker="o",
        markersize=3.5,
        label="GraphCast — Native",
    )

    ax.plot(
        gc_wuduan[
            "forecast_longitude_plot"
        ],
        gc_wuduan[
            "forecast_latitude"
        ],
        linestyle="-",
        linewidth=2.0,
        marker="s",
        markersize=3.5,
        label="GraphCast — WuDuan",
    )

    # --------------------------------------------------------
    # AIFS2
    # --------------------------------------------------------

    ax.plot(
        ai_native[
            "forecast_longitude_plot"
        ],
        ai_native[
            "forecast_latitude"
        ],
        linestyle="--",
        linewidth=1.8,
        marker="^",
        markersize=3.5,
        label="AIFS2 — Native",
    )

    ax.plot(
        ai_wuduan[
            "forecast_longitude_plot"
        ],
        ai_wuduan[
            "forecast_latitude"
        ],
        linestyle="-",
        linewidth=2.0,
        marker="D",
        markersize=3.5,
        label="AIFS2 — WuDuan",
    )

    # --------------------------------------------------------
    # Selected lead-time labels
    #
    # Use fewer labels here because five trajectories are shown.
    # --------------------------------------------------------

    for data in [
        gc_native,
        gc_wuduan,
        ai_native,
        ai_wuduan,
    ]:

        selected = data[
            data[
                "lead_time_hours"
            ].isin(
                [
                    48,
                    96,
                    144,
                    192,
                    240,
                ]
            )
        ]

        for _, row in selected.iterrows():

            ax.annotate(
                f"{int(row['lead_time_hours'])}",
                (
                    row[
                        "forecast_longitude_plot"
                    ],
                    row[
                        "forecast_latitude"
                    ],
                ),
                xytext=(
                    4,
                    4,
                ),
                textcoords="offset points",
                fontsize=6.5,
                alpha=0.8,
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

axes[0].set_ylabel(
    "Latitude (°)"
)

axes[0].legend(
    loc="best",
    fontsize=8,
)

fig.suptitle(
    "Eastern Pacific Tropical Cyclone Tracks\n"
    "IBTrACS, Native, and WuDuan — GraphCast vs AIFS2",
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

all_tracks_png, all_tracks_pdf = save_figure(
    fig,
    "graphcast_vs_aifs2_all_tracks_by_storm",
)


# ============================================================
# FIGURE 5
# Native track error by storm
# ============================================================

fig, axes = plt.subplots(
    1,
    4,
    figsize=(20, 5.5),
    sharey=True,
)

for ax, (
    case_id,
    storm_name,
) in zip(
    axes,
    STORMS,
):

    gc = storm_subset(
        graphcast_native,
        case_id,
    )

    ai = storm_subset(
        aifs2_native,
        case_id,
    )

    ax.plot(
        gc[
            "lead_time_hours"
        ],
        gc[
            "track_error_km"
        ],
        marker="o",
        markersize=4,
        linewidth=2,
        label="GraphCast Native",
    )

    ax.plot(
        ai[
            "lead_time_hours"
        ],
        ai[
            "track_error_km"
        ],
        marker="s",
        markersize=4,
        linewidth=2,
        label="AIFS2 Native",
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

axes[0].set_ylabel(
    "Track error (km)"
)

axes[0].legend(
    loc="best",
)

fig.suptitle(
    "Tropical Cyclone Track Error by Lead Time\n"
    "GraphCast vs AIFS2 — Native Tracker",
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

native_error_png, native_error_pdf = save_figure(
    fig,
    "graphcast_vs_aifs2_native_track_error_by_storm",
)


# ============================================================
# FIGURE 6
# Tracker sensitivity
#
# Rows:
#     GraphCast
#     AIFS2
#
# Columns:
#     Elida
#     Fausto
#     Genevieve
#     Hernan
# ============================================================

fig, axes = plt.subplots(
    2,
    4,
    figsize=(20, 9),
    sharex=True,
    sharey="row",
)

model_rows = [
    (
        "GraphCast",
        graphcast_native,
        graphcast_wuduan,
    ),
    (
        "AIFS2",
        aifs2_native,
        aifs2_wuduan,
    ),
]

for row_index, (
    model_label,
    native_data,
    wuduan_data,
) in enumerate(
    model_rows
):

    for column_index, (
        case_id,
        storm_name,
    ) in enumerate(
        STORMS
    ):

        ax = axes[
            row_index,
            column_index,
        ]

        native = storm_subset(
            native_data,
            case_id,
        )

        wuduan = storm_subset(
            wuduan_data,
            case_id,
        )

        ax.plot(
            native[
                "lead_time_hours"
            ],
            native[
                "track_error_km"
            ],
            marker="o",
            markersize=3.5,
            linewidth=1.8,
            linestyle="--",
            label="Native",
        )

        ax.plot(
            wuduan[
                "lead_time_hours"
            ],
            wuduan[
                "track_error_km"
            ],
            marker="s",
            markersize=3.5,
            linewidth=2,
            linestyle="-",
            label="WuDuan",
        )

        if native.empty:
            ax.text(
                0.5,
                0.64,
                "No native track",
                transform=ax.transAxes,
                ha="center",
                va="center",
                fontsize=9.5,
                fontstyle="italic",
                alpha=0.7,
            )

        if wuduan.empty:
            ax.text(
                0.5,
                0.42,
                "No accepted WuDuan track",
                transform=ax.transAxes,
                ha="center",
                va="center",
                fontsize=9.5,
                fontstyle="italic",
                alpha=0.7,
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

        if row_index == 0:
            ax.set_title(
                storm_name,
                fontweight="bold",
            )

        if row_index == 1:
            ax.set_xlabel(
                "Forecast lead time (h)"
            )

        if column_index == 0:
            ax.set_ylabel(
                f"{model_label}\n"
                "Track error (km)"
            )

axes[0, 0].legend(
    loc="best",
)

fig.suptitle(
    "Tropical Cyclone Tracker Sensitivity\n"
    "Native vs WuDuan for GraphCast and AIFS2",
    fontsize=14,
    fontweight="bold",
)

fig.tight_layout(
    rect=[
        0,
        0,
        1,
        0.94,
    ]
)

sensitivity_png, sensitivity_pdf = save_figure(
    fig,
    "native_vs_wuduan_track_error_by_model_storm",
)


# ============================================================
# Console summary
# ============================================================

print("=" * 88)
print(
    "GRAPHCAST vs AIFS2 — "
    "PAIRED WUDUAN TRACK COMPARISON"
)
print("=" * 88)

print()

display = summary.copy()

if not display.empty:

    display[
        "aifs2_better_percent"
    ] = (
        display[
            "aifs2_better_fraction"
        ]
        * 100.0
    )

    columns = [
        "storm",
        "paired_points",
        "graphcast_mean_error_km",
        "aifs2_mean_error_km",
        "graphcast_rmse_km",
        "aifs2_rmse_km",
        "mean_delta_error_km",
        "aifs2_better_points",
        "graphcast_better_points",
        "aifs2_better_percent",
    ]

    print(
        display[
            columns
        ].to_string(
            index=False,
            float_format=lambda x:
                f"{x:.1f}",
        )
    )


print()
print("=" * 88)
print(
    "TRACKER SENSITIVITY — "
    "FULL COVERAGE"
)
print("=" * 88)

print()

if not tracker_summary.empty:

    tracker_display = (
        tracker_summary[
            [
                "storm",
                "model_name",
                "tracker",
                "point_count",
                "mean_track_error_km",
                "rmse_track_error_km",
                "maximum_track_error_km",
                "last_lead_time_hours",
            ]
        ]
        .sort_values(
            [
                "storm",
                "model_name",
                "tracker",
            ]
        )
    )

    print(
        tracker_display.to_string(
            index=False,
            float_format=lambda x:
                f"{x:.1f}",
        )
    )


print()
print("=" * 88)
print("OUTPUT PRODUCTS")
print("=" * 88)

print()
print("Paired WuDuan table:")
print(
    paired_path
)

print()
print("Paired WuDuan summary:")
print(
    summary_path
)

print()
print("Tracker sensitivity summary:")
print(
    tracker_summary_path
)

print()
print("WuDuan figures:")
print(
    track_png
)
print(
    error_png
)
print(
    delta_png
)

print()
print("Native figures:")
print(
    native_track_png
)
print(
    native_error_png
)

print()
print("All-track comparison:")
print(
    all_tracks_png
)

print()
print("Tracker-sensitivity figure:")
print(
    sensitivity_png
)