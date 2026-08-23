from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ============================================================
# Configuration
# ============================================================

STORMS = {
    "Elida": {
        "diagnostics": Path(
            "results/waves/aifs2_elida/"
            "aifs2_elida_wave_diagnostics.csv"
        ),
        "radial": Path(
            "results/waves/aifs2_elida/"
            "aifs2_elida_wave_radial_profiles.csv"
        ),
    },
    "Fausto": {
        "diagnostics": Path(
            "results/waves/aifs2_fausto/"
            "aifs2_fausto_wave_diagnostics.csv"
        ),
        "radial": Path(
            "results/waves/aifs2_fausto/"
            "aifs2_fausto_wave_radial_profiles.csv"
        ),
    },
}

OUTPUT_DIR = Path(
    "results/waves/aifs2_storm_comparison"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# Helpers
# ============================================================

def normalize(series):

    values = np.asarray(
        series,
        dtype=float,
    )

    maximum = np.nanmax(
        values
    )

    if (
        not np.isfinite(maximum)
        or maximum == 0.0
    ):
        return np.full_like(
            values,
            np.nan,
        )

    return (
        values
        / maximum
    )


# ============================================================
# Read data
# ============================================================

diagnostics = {}

radial_profiles = {}

for storm, paths in STORMS.items():

    diagnostics[
        storm
    ] = (
        pd.read_csv(
            paths[
                "diagnostics"
            ]
        )
        .sort_values(
            "lead_time_hours"
        )
        .reset_index(
            drop=True
        )
    )

    radial_profiles[
        storm
    ] = (
        pd.read_csv(
            paths[
                "radial"
            ]
        )
        .sort_values(
            [
                "lead_time_hours",
                "radius_km",
            ]
        )
        .reset_index(
            drop=True
        )
    )


# ============================================================
# Storm summaries
# ============================================================

summary_rows = []

ridge_tables = {}


for storm in STORMS:

    diag = diagnostics[
        storm
    ]

    radial = radial_profiles[
        storm
    ]


    # --------------------------------------------------------
    # Maximum pointwise SWH within 300 km
    # --------------------------------------------------------

    swh_index = (
        diag[
            "max_swh_300km_m"
        ].idxmax()
    )

    swh_peak = float(
        diag.loc[
            swh_index,
            "max_swh_300km_m",
        ]
    )

    swh_peak_time = int(
        diag.loc[
            swh_index,
            "lead_time_hours",
        ]
    )


    # --------------------------------------------------------
    # Maximum pointwise wind within 300 km
    # --------------------------------------------------------

    wind_index = (
        diag[
            "max_wind_300km_ms"
        ].idxmax()
    )

    wind_peak = float(
        diag.loc[
            wind_index,
            "max_wind_300km_ms",
        ]
    )

    wind_peak_time = int(
        diag.loc[
            wind_index,
            "lead_time_hours",
        ]
    )


    # --------------------------------------------------------
    # Radius of maximum radial-mean SWH
    # --------------------------------------------------------

    ridge = (
        radial[
            radial[
                "is_max_swh_radius"
            ].astype(bool)
        ]
        .copy()
        .sort_values(
            "lead_time_hours"
        )
    )

    ridge_tables[
        storm
    ] = ridge


    radial_peak_index = (
        ridge[
            "max_radial_mean_swh_m"
        ].idxmax()
    )

    radial_peak_swh = float(
        ridge.loc[
            radial_peak_index,
            "max_radial_mean_swh_m",
        ]
    )

    radial_peak_time = int(
        ridge.loc[
            radial_peak_index,
            "lead_time_hours",
        ]
    )

    radial_peak_radius = float(
        ridge.loc[
            radial_peak_index,
            "radius_km",
        ]
    )


    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    summary_rows.append(
        {
            "storm":
                storm,

            "max_wind_300km_ms":
                wind_peak,

            "wind_peak_lead_h":
                wind_peak_time,

            "max_swh_300km_m":
                swh_peak,

            "swh_peak_lead_h":
                swh_peak_time,

            "wind_to_swh_lag_h":
                (
                    swh_peak_time
                    - wind_peak_time
                ),

            "max_radial_mean_swh_m":
                radial_peak_swh,

            "radial_mean_peak_lead_h":
                radial_peak_time,

            "radial_mean_peak_radius_km":
                radial_peak_radius,
        }
    )


summary = pd.DataFrame(
    summary_rows
)


# ============================================================
# Export summary
# ============================================================

summary_path = (
    OUTPUT_DIR
    / "aifs2_elida_fausto_wave_summary.csv"
)

summary.to_csv(
    summary_path,
    index=False,
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
    sharex=False,
)


# ============================================================
# Panel 1
# Normalized maximum wind and SWH
# ============================================================

ax = axes[0]

for storm in STORMS:

    diag = diagnostics[
        storm
    ]

    lead = diag[
        "lead_time_hours"
    ].values

    wind_norm = normalize(
        diag[
            "max_wind_300km_ms"
        ].values
    )

    swh_norm = normalize(
        diag[
            "max_swh_300km_m"
        ].values
    )


    ax.plot(
        lead,
        wind_norm,
        linewidth=2,
        marker="o",
        markersize=3,
        label=f"{storm} wind",
    )

    ax.plot(
        lead,
        swh_norm,
        linewidth=2,
        linestyle="--",
        label=f"{storm} SWH",
    )


ax.set_ylabel(
    "Normalized amplitude"
)

ax.set_title(
    "Storm-relative wind and wave evolution\n"
    "Maximum values within 300 km"
)

ax.set_ylim(
    0.0,
    1.08,
)

ax.grid(
    True,
    alpha=0.3,
)

ax.legend(
    ncol=2,
    fontsize=9,
)


# ============================================================
# Panel 2
# Radius of maximum radial-mean SWH
# ============================================================

ax = axes[1]

for storm in STORMS:

    ridge = ridge_tables[
        storm
    ]

    ax.plot(
        ridge[
            "lead_time_hours"
        ],
        ridge[
            "radius_km"
        ],
        linewidth=2,
        marker="o",
        markersize=4,
        label=storm,
    )


ax.set_xlabel(
    "Forecast lead time (h)"
)

ax.set_ylabel(
    "Radius of maximum radial-mean SWH (km)"
)

ax.set_title(
    "Radial position of strongest azimuthal-mean wave field"
)

ax.set_ylim(
    0,
    200,
)

ax.grid(
    True,
    alpha=0.3,
)

ax.legend()


# ============================================================
# Panel 3
# Peak timing / lag
# ============================================================

ax = axes[2]

x = np.arange(
    len(summary)
)

width = 0.34


ax.bar(
    x - width / 2.0,
    summary[
        "wind_peak_lead_h"
    ],
    width,
    label="Wind peak",
)

ax.bar(
    x + width / 2.0,
    summary[
        "swh_peak_lead_h"
    ],
    width,
    label="SWH peak",
)


for index, row in summary.iterrows():

    ymax = max(
        row[
            "wind_peak_lead_h"
        ],
        row[
            "swh_peak_lead_h"
        ],
    )

    ax.text(
        index,
        ymax + 5,
        (
            f"Lag = "
            f"{int(row['wind_to_swh_lag_h'])} h"
        ),
        ha="center",
        va="bottom",
        fontsize=9,
    )


ax.set_xticks(
    x
)

ax.set_xticklabels(
    summary[
        "storm"
    ]
)

ax.set_ylabel(
    "Forecast lead time (h)"
)

ax.set_title(
    "Timing of maximum 300-km wind and SWH"
)

ax.grid(
    True,
    axis="y",
    alpha=0.3,
)

ax.legend()


# ============================================================
# Overall title
# ============================================================

fig.suptitle(
    "AIFS2 Tropical-Cyclone Wave Comparison\n"
    "Elida vs Fausto — WuDuan storm-relative diagnostics",
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
# Save figure
# ============================================================

png_path = (
    OUTPUT_DIR
    / "aifs2_elida_vs_fausto_wave_comparison.png"
)

pdf_path = (
    OUTPUT_DIR
    / "aifs2_elida_vs_fausto_wave_comparison.pdf"
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
# Console output
# ============================================================

print("=" * 82)
print(
    "AIFS2 TROPICAL-CYCLONE WAVE COMPARISON"
)
print("=" * 82)

print()

print(
    summary.to_string(
        index=False,
        float_format=lambda x: f"{x:.2f}",
    )
)

print()

print(
    "Summary:",
    summary_path,
)

print(
    "Figure:",
    png_path,
)

print(
    "PDF:",
    pdf_path,
)