"""
Integrated AIFS2 tropical-cyclone structure and wave comparison.

Compares the early Douglas and Iselle forecasts using:

1. AIFS2 pre-genesis radial organized-vortex diagnostic.
2. Standard AIWeather TC verification outcome.
3. IBTrACS-centered AIFS2 wave diagnostics.
4. Radius of maximum radial-mean significant wave height.

The observed TD/TS milestones below are case metadata established
during the Douglas/Iselle case analysis. They are not inferred from
the verification ibtracs.csv files, which do not contain storm status.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# Configuration
# ============================================================

OUTPUT_DIR = Path(
    "results/waves/"
    "aifs2_douglas_iselle_comparison"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


CASES = {
    "douglas": {
        "storm_name": "Douglas",
        "init_label": "2026-06-29 00 UTC",
        "radial_path": Path(
            "results/verification/comparison/"
            "douglas_pregenesis/"
            "douglas_20260629T000000_radial_diagnostic.csv"
        ),
        "verification_path": Path(
            "results/verification/"
            "aifs2_douglas_20260629T000000/"
            "verification_summary.csv"
        ),
        "wave_path": Path(
            "results/waves/"
            "aifs2_douglas/"
            "aifs2_douglas_wave_diagnostics.csv"
        ),
        "wave_radial_path": Path(
            "results/waves/"
            "aifs2_douglas/"
            "aifs2_douglas_wave_radial_profiles.csv"
        ),
        "milestones": [
            ("Observed TD", 54),
            ("Observed TS", 66),
        ],
    },

    "iselle": {
        "storm_name": "Iselle",
        "init_label": "2026-08-21 18 UTC",
        "radial_path": Path(
            "results/verification/comparison/"
            "iselle_pregenesis/"
            "iselle_20260821T180000_radial_diagnostic.csv"
        ),
        "verification_path": Path(
            "results/verification/"
            "aifs2_iselle_20260821T180000/"
            "verification_summary.csv"
        ),
        "wave_path": Path(
            "results/waves/"
            "aifs2_iselle/"
            "aifs2_iselle_wave_diagnostics.csv"
        ),
        "wave_radial_path": Path(
            "results/waves/"
            "aifs2_iselle/"
            "aifs2_iselle_wave_radial_profiles.csv"
        ),
        "milestones": [
            ("Observed TS", 42),
        ],
    },
}


PLOT_MAX_LEAD = 120


# ============================================================
# Helpers
# ============================================================

def first_organized_vortex(radial):
    """
    Return first AIFS2 lead satisfying organized-vortex criteria.
    """

    aifs2 = radial[
        radial["model"].astype(str).str.lower()
        == "aifs2"
    ].copy()

    organized = (
        aifs2["organized_vortex"]
        .astype(str)
        .str.lower()
        .eq("true")
    )

    hits = aifs2[
        organized
    ].sort_values(
        "lead_time_hours"
    )

    if hits.empty:
        return np.nan

    return float(
        hits.iloc[0][
            "lead_time_hours"
        ]
    )


def radial_swh_maximum_radius(radial):
    """
    Return one radius of maximum radial-mean SWH per lead.
    """

    radial = radial.copy()

    idx = (
        radial
        .groupby(
            "lead_time_hours"
        )["mean_swh_m"]
        .idxmax()
    )

    return (
        radial.loc[
            idx,
            [
                "lead_time_hours",
                "radius_km",
                "mean_swh_m",
            ],
        ]
        .sort_values(
            "lead_time_hours"
        )
        .reset_index(
            drop=True
        )
    )


def verification_description(verification):
    """
    Create concise native-tracker verification description.
    """

    full = verification[
        (
            verification["tracker"]
            == "native"
        )
        & (
            verification["coverage"]
            == "full"
        )
    ]

    if full.empty:
        return (
            "Native verification unavailable"
        )

    row = full.iloc[0]

    n = int(
        row["overlap_count"]
    )

    if n == 0:
        return (
            "No accepted native track "
            "(overlap n=0)"
        )

    mean_error = row[
        "mean_track_error_km"
    ]

    return (
        f"Native overlap n={n}; "
        f"mean error={mean_error:.1f} km"
    )


def add_event_lines(
    ax,
    milestones,
    organized_lead,
):
    """
    Add observed-development and organized-vortex markers.
    """

    line_styles = [
        "--",
        "-.",
        ":",
    ]

    for index, (
        label,
        lead,
    ) in enumerate(
        milestones
    ):

        ax.axvline(
            lead,
            linestyle=line_styles[
                index
                % len(line_styles)
            ],
            linewidth=1.2,
            alpha=0.8,
            label=label,
        )

    if np.isfinite(
        organized_lead
    ):
        ax.axvline(
            organized_lead,
            linestyle="-",
            linewidth=1.5,
            alpha=0.9,
            label=(
                "AIFS2 first "
                "organized vortex"
            ),
        )


def shade_unavailable_window(
    ax,
    final_lead,
):
    """
    Shade forecast time beyond IBTrACS-centered wave coverage.
    """

    if final_lead >= PLOT_MAX_LEAD:
        return

    ax.axvspan(
        final_lead,
        PLOT_MAX_LEAD,
        alpha=0.10,
    )


# ============================================================
# Load cases
# ============================================================

loaded = {}
summary_rows = []


for key, config in CASES.items():

    radial = pd.read_csv(
        config[
            "radial_path"
        ]
    )

    verification = pd.read_csv(
        config[
            "verification_path"
        ]
    )

    wave = pd.read_csv(
        config[
            "wave_path"
        ]
    )

    wave_radial = pd.read_csv(
        config[
            "wave_radial_path"
        ]
    )

    organized_lead = (
        first_organized_vortex(
            radial
        )
    )

    radius = (
        radial_swh_maximum_radius(
            wave_radial
        )
    )

    tracker_text = (
        verification_description(
            verification
        )
    )

    final_wave_lead = float(
        wave[
            "lead_time_hours"
        ].max()
    )

    max_swh_index = (
        wave[
            "max_swh_300km_m"
        ].idxmax()
    )

    max_wind_index = (
        wave[
            "max_wind_300km_ms"
        ].idxmax()
    )

    max_swh_row = wave.loc[
        max_swh_index
    ]

    max_wind_row = wave.loc[
        max_wind_index
    ]

    max_swh_censored = (
        float(
            max_swh_row[
                "lead_time_hours"
            ]
        )
        == final_wave_lead
    )

    max_wind_censored = (
        float(
            max_wind_row[
                "lead_time_hours"
            ]
        )
        == final_wave_lead
    )

    radial_peak_index = (
        wave_radial[
            "mean_swh_m"
        ].idxmax()
    )

    radial_peak = (
        wave_radial.loc[
            radial_peak_index
        ]
    )

    loaded[key] = {
        "config": config,
        "radial": radial,
        "verification": verification,
        "wave": wave,
        "radius": radius,
        "organized_lead":
            organized_lead,
        "tracker_text":
            tracker_text,
        "final_wave_lead":
            final_wave_lead,
    }

    milestone_map = {
        label: lead
        for label, lead
        in config["milestones"]
    }

    summary_rows.append(
        {
            "storm":
                config[
                    "storm_name"
                ],
            "initialization":
                config[
                    "init_label"
                ],
            "observed_td_lead_h":
                milestone_map.get(
                    "Observed TD",
                    np.nan,
                ),
            "observed_ts_lead_h":
                milestone_map.get(
                    "Observed TS",
                    np.nan,
                ),
            "center_source":
                "IBTrACS",
            "first_organized_vortex_h":
                organized_lead,
            "wave_window_end_h":
                final_wave_lead,
            "max_300km_swh_m":
                max_swh_row[
                    "max_swh_300km_m"
                ],
            "max_300km_swh_lead_h":
                max_swh_row[
                    "lead_time_hours"
                ],
            "max_300km_swh_censored":
                max_swh_censored,
            "max_300km_wind_ms":
                max_wind_row[
                    "max_wind_300km_ms"
                ],
            "max_300km_wind_lead_h":
                max_wind_row[
                    "lead_time_hours"
                ],
            "max_300km_wind_censored":
                max_wind_censored,
            "max_radial_mean_swh_m":
                radial_peak[
                    "mean_swh_m"
                ],
            "max_radial_mean_swh_lead_h":
                radial_peak[
                    "lead_time_hours"
                ],
            "max_radial_mean_swh_radius_km":
                radial_peak[
                    "radius_km"
                ],
            "verification":
                tracker_text,
        }
    )


summary = pd.DataFrame(
    summary_rows
)

summary_path = (
    OUTPUT_DIR
    / "aifs2_douglas_iselle_summary.csv"
)

summary.to_csv(
    summary_path,
    index=False,
)


# ============================================================
# Figure
# ============================================================

fig, axes = plt.subplots(
    nrows=3,
    ncols=2,
    figsize=(15, 11),
    sharex=True,
)


for column, key in enumerate(
    [
        "douglas",
        "iselle",
    ]
):

    case = loaded[
        key
    ]

    config = case[
        "config"
    ]

    wave = case[
        "wave"
    ]

    radius = case[
        "radius"
    ]

    organized_lead = case[
        "organized_lead"
    ]

    final_wave_lead = case[
        "final_wave_lead"
    ]

    # --------------------------------------------------------
    # SWH
    # --------------------------------------------------------

    ax = axes[
        0,
        column,
    ]

    ax.plot(
        wave[
            "lead_time_hours"
        ],
        wave[
            "mean_swh_300km_m"
        ],
        marker="o",
        label="300-km mean SWH",
    )

    ax.plot(
        wave[
            "lead_time_hours"
        ],
        wave[
            "max_swh_300km_m"
        ],
        marker="s",
        label="300-km max SWH",
    )

    terminal_swh = wave.iloc[-1]

    if (
        terminal_swh[
            "max_swh_300km_m"
        ]
        == wave[
            "max_swh_300km_m"
        ].max()
    ):
        ax.annotate(
            "*",
            (
                terminal_swh[
                    "lead_time_hours"
                ],
                terminal_swh[
                    "max_swh_300km_m"
                ],
            ),
            xytext=(5, 4),
            textcoords="offset points",
            fontsize=14,
            fontweight="bold",
        )

    add_event_lines(
        ax,
        config[
            "milestones"
        ],
        organized_lead,
    )

    shade_unavailable_window(
        ax,
        final_wave_lead,
    )

    ax.set_ylabel(
        "SWH (m)"
    )

    ax.set_title(
        (
            f"{config['storm_name']} — "
            f"{config['init_label']}\n"
            f"{case['tracker_text']}"
        )
    )

    ax.grid(
        alpha=0.25
    )

    ax.legend(
        fontsize=8,
        loc="best",
    )

    # --------------------------------------------------------
    # Wind
    # --------------------------------------------------------

    ax = axes[
        1,
        column,
    ]

    ax.plot(
        wave[
            "lead_time_hours"
        ],
        wave[
            "max_wind_300km_ms"
        ],
        marker="o",
        label="300-km max wind",
    )

    terminal_wind = wave.iloc[-1]

    if (
        terminal_wind[
            "max_wind_300km_ms"
        ]
        == wave[
            "max_wind_300km_ms"
        ].max()
    ):
        ax.annotate(
            "*",
            (
                terminal_wind[
                    "lead_time_hours"
                ],
                terminal_wind[
                    "max_wind_300km_ms"
                ],
            ),
            xytext=(5, 4),
            textcoords="offset points",
            fontsize=14,
            fontweight="bold",
        )

    add_event_lines(
        ax,
        config[
            "milestones"
        ],
        organized_lead,
    )

    shade_unavailable_window(
        ax,
        final_wave_lead,
    )

    ax.set_ylabel(
        "Wind speed (m s$^{-1}$)"
    )

    ax.grid(
        alpha=0.25
    )

    # --------------------------------------------------------
    # Radius of maximum radial-mean SWH
    # --------------------------------------------------------

    ax = axes[
        2,
        column,
    ]

    ax.plot(
        radius[
            "lead_time_hours"
        ],
        radius[
            "radius_km"
        ],
        marker="o",
        label=(
            r"$R_{\mathrm{SWHmax}}$"
        ),
    )

    add_event_lines(
        ax,
        config[
            "milestones"
        ],
        organized_lead,
    )

    shade_unavailable_window(
        ax,
        final_wave_lead,
    )

    ax.set_ylabel(
        (
            "Radius of maximum\n"
            "radial-mean SWH (km)"
        )
    )

    ax.set_xlabel(
        "Forecast lead time (h)"
    )

    ax.grid(
        alpha=0.25
    )


for ax in axes.flat:

    ax.set_xlim(
        0,
        PLOT_MAX_LEAD,
    )

    ax.set_xticks(
        np.arange(
            0,
            PLOT_MAX_LEAD + 1,
            12,
        )
    )


fig.suptitle(
    (
        "AIFS2 Tropical-Cyclone Structure and "
        "Storm-Relative Wave Evolution\n"
        "IBTrACS-centered Douglas and Iselle comparison"
    ),
    fontsize=15,
    fontweight="bold",
)


fig.text(
    0.5,
    0.015,
    (
        "Shaded interval: no IBTrACS center available for "
        "storm-relative wave diagnostics. "
        "* Maximum occurs at final available center and is right-censored."
    ),
    ha="center",
    fontsize=9,
)


fig.tight_layout(
    rect=[
        0,
        0.04,
        1,
        0.94,
    ]
)


png_path = (
    OUTPUT_DIR
    / "aifs2_douglas_iselle_structure_waves.png"
)

pdf_path = (
    OUTPUT_DIR
    / "aifs2_douglas_iselle_structure_waves.pdf"
)


fig.savefig(
    png_path,
    dpi=200,
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

print(
    "=" * 78
)

print(
    "AIFS2 DOUGLAS–ISELLE "
    "STRUCTURE/WAVE COMPARISON"
)

print(
    "=" * 78
)

print()

print(
    summary.to_string(
        index=False
    )
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

print(
    summary_path
)
