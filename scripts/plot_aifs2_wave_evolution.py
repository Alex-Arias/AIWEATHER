from pathlib import Path
import argparse

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

from wave_cases import get_case


# ============================================================
# Command-line configuration
# ============================================================

parser = argparse.ArgumentParser(
    description=(
        "Plot AIFS2 storm-relative wave evolution."
    )
)

parser.add_argument(
    "--storm",
    required=True,
    help=(
        "Storm key defined in scripts/wave_cases.py "
        "(for example: elida, fausto, genevieve)."
    ),
)

args = parser.parse_args()

CASE = get_case(
    args.storm
)

STORM_KEY = args.storm.lower()

STORM_NAME = CASE[
    "storm_name"
]


# ============================================================
# Paths
# ============================================================

OUTPUT_DIR = Path(
    "results/waves"
) / f"aifs2_{STORM_KEY}"

INPUT_PATH = (
    OUTPUT_DIR
    / f"aifs2_{STORM_KEY}_wave_diagnostics.csv"
)

REFERENCE_LEADS = [
    24,
    48,
    72,
    96,
    120,
]

RADII = [
    300,
    500,
    800,
]

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# Read diagnostics
# ============================================================

df = pd.read_csv(
    INPUT_PATH
)

df = df.sort_values(
    "lead_time_hours"
).reset_index(
    drop=True
)


# ============================================================
# Figure
# ============================================================

fig, axes = plt.subplots(
    3,
    1,
    figsize=(
        11,
        11,
    ),
    sharex=True,
)


# ============================================================
# Panel 1
# Maximum and mean SWH
# ============================================================

ax = axes[0]

for radius in RADII:

    ax.plot(
        df[
            "lead_time_hours"
        ],
        df[
            f"max_swh_{radius}km_m"
        ],
        marker="o",
        markersize=3.5,
        linewidth=1.8,
        label=(
            f"Max SWH ≤ {radius} km"
        ),
    )

    ax.plot(
        df[
            "lead_time_hours"
        ],
        df[
            f"mean_swh_{radius}km_m"
        ],
        linestyle="--",
        linewidth=1.4,
        label=(
            f"Mean SWH ≤ {radius} km"
        ),
    )

ax.set_ylabel(
    "SWH (m)"
)

ax.set_title(
    "Storm-relative significant wave height"
)

ax.grid(
    True,
    alpha=0.3,
)

ax.legend(
    ncol=2,
    fontsize=8,
)


# ============================================================
# Panel 2
# Wind and SWH
# ============================================================

ax = axes[1]

ax.plot(
    df[
        "lead_time_hours"
    ],
    df[
        "max_wind_300km_ms"
    ],
    marker="o",
    markersize=4,
    linewidth=2,
    label="Maximum 10-m wind ≤ 300 km",
)

ax.set_ylabel(
    "Wind speed (m s$^{-1}$)"
)

ax.set_title(
    "Storm-relative wind forcing"
)

ax.grid(
    True,
    alpha=0.3,
)

ax.legend(
    loc="upper left",
)


# Optional second y-axis for maximum SWH
ax2 = ax.twinx()

ax2.plot(
    df[
        "lead_time_hours"
    ],
    df[
        "max_swh_300km_m"
    ],
    linestyle="--",
    linewidth=2,
    label="Maximum SWH ≤ 300 km",
)

ax2.set_ylabel(
    "SWH (m)"
)

handles1, labels1 = (
    ax.get_legend_handles_labels()
)

handles2, labels2 = (
    ax2.get_legend_handles_labels()
)

ax.legend(
    handles1 + handles2,
    labels1 + labels2,
    loc="upper left",
    fontsize=8,
)


# ============================================================
# Panel 3
# Mean wave period
# ============================================================

ax = axes[2]

for radius in RADII:

    ax.plot(
        df[
            "lead_time_hours"
        ],
        df[
            f"mean_mwp_{radius}km_s"
        ],
        marker="o",
        markersize=3.5,
        linewidth=1.8,
        label=(
            f"Mean MWP ≤ {radius} km"
        ),
    )

ax.set_xlabel(
    "Forecast lead time (h)"
)

ax.set_ylabel(
    "Mean wave period (s)"
)

ax.set_title(
    "Storm-relative mean wave period"
)

ax.grid(
    True,
    alpha=0.3,
)

ax.legend(
    fontsize=8,
)


# ============================================================
# Reference lead times
# ============================================================

for ax in axes:

    for lead in REFERENCE_LEADS:

        ax.axvline(
            lead,
            linestyle=":",
            linewidth=0.8,
            alpha=0.45,
        )


# ============================================================
# Title and layout
# ============================================================

fig.suptitle(
    f"AIFS2 Tropical-Cyclone Wave Evolution — {STORM_NAME}\n"
    "WuDuan storm-relative diagnostics",
    fontsize=14,
    fontweight="bold",
)

fig.tight_layout(
    rect=[
        0,
        0,
        1,
        0.95,
    ]
)


# ============================================================
# Save
# ============================================================

png_path = (
    OUTPUT_DIR
    / (
        f"aifs2_{STORM_NAME.lower()}_"
        "wave_evolution.png"
    )
)

pdf_path = (
    OUTPUT_DIR
    / (
        f"aifs2_{STORM_NAME.lower()}_"
        "wave_evolution.pdf"
    )
)

fig.savefig(
    png_path,
    dpi=300,
    bbox_inches="tight",
)

fig.savefig(
    pdf_path,
    bbox_inches="tight",
)

plt.close(
    fig
)


# ============================================================
# Console summary
# ============================================================

print("=" * 78)
print(
    f"AIFS2 {STORM_NAME.upper()} "
    "WAVE-EVOLUTION FIGURE"
)
print("=" * 78)

print()
print(
    "Input:",
    INPUT_PATH,
)

print()
print(
    "Saved:"
)

print(
    png_path
)

print(
    pdf_path
)

print()
print(
    "Peak values:"
)

peak_swh_index = (
    df[
        "max_swh_300km_m"
    ].idxmax()
)

peak_wind_index = (
    df[
        "max_wind_300km_ms"
    ].idxmax()
)

print(
    "Maximum 300-km SWH:",
    f"{df.loc[peak_swh_index, 'max_swh_300km_m']:.2f} m",
    "at",
    f"+{int(df.loc[peak_swh_index, 'lead_time_hours'])} h",
)

print(
    "Maximum 300-km wind:",
    f"{df.loc[peak_wind_index, 'max_wind_300km_ms']:.2f} m/s",
    "at",
    f"+{int(df.loc[peak_wind_index, 'lead_time_hours'])} h",
)