"""
Final summary figure for the Odalys–Polo AIWeather operational experiment.

Experiment:
    Cycles 1–9 : accepted operational tracking population
    Cycle 10   : termination; no accepted tracker continuation

Panels:
    A. Tracker detection fraction through Cycles 1–10.
    B. Odalys WuDuan historical vs recent-3-cycle variability.
    C. Polo WuDuan historical vs recent-3-cycle variability.

The variability panels use only rows where comparison_valid=True for
contraction diagnostics.
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path("results/operational")
OUT = Path("outputs/verification")
OUT.mkdir(parents=True, exist_ok=True)

DETECTION = ROOT / "odalys_polo_cycles1-10_detection_matrix.csv"

VARIABILITY = {
    "Odalys": (
        ROOT
        / "odalys_multicycle"
        / "odalys_wuduan_variability_comparison.csv"
    ),
    "Polo": (
        ROOT
        / "polo_multicycle"
        / "polo_wuduan_variability_comparison.csv"
    ),
}


# ---------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------

det = pd.read_csv(DETECTION)

required_det = {
    "cycle",
    "init",
    "storm",
    "model",
    "tracker",
    "detected",
}
missing = required_det - set(det.columns)
if missing:
    raise ValueError(
        f"Detection matrix missing columns: {sorted(missing)}"
    )

det["detected"] = det["detected"].astype(bool)

var = {}

for storm, path in VARIABILITY.items():

    df = pd.read_csv(path)
    df["valid_time"] = pd.to_datetime(df["valid_time"])

    required = {
        "valid_time",
        "historical_r67_km",
        "historical_r90_km",
        "recent_r67_km",
        "recent_r90_km",
        "r67_spread_contraction_pct",
        "r90_spread_contraction_pct",
        "comparison_valid",
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"{storm} variability file missing "
            f"{sorted(missing)}"
        )

    # Robust conversion in case CSV stores booleans as strings.
    if df["comparison_valid"].dtype != bool:
        df["comparison_valid"] = (
            df["comparison_valid"]
            .astype(str)
            .str.lower()
            .map({"true": True, "false": False})
            .fillna(False)
        )

    var[storm] = df


# ---------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------

fig = plt.figure(figsize=(16, 13))

gs = fig.add_gridspec(
    3,
    1,
    height_ratios=[1.05, 1.0, 1.0],
    hspace=0.34,
)

ax0 = fig.add_subplot(gs[0])
ax1 = fig.add_subplot(gs[1])
ax2 = fig.add_subplot(gs[2])


# ---------------------------------------------------------------------
# Panel A — detection fraction
# ---------------------------------------------------------------------

trackers = ["native", "wuduan", "vitart"]
storms = ["Odalys", "Polo"]

linestyles = {
    "Odalys": "-",
    "Polo": "--",
}

markers = {
    "native": "o",
    "wuduan": "s",
    "vitart": "^",
}

for storm in storms:

    subset = det[det["storm"] == storm]

    for tracker in trackers:

        x = (
            subset[subset["tracker"] == tracker]
            .groupby("cycle")["detected"]
            .mean()
            .reindex(range(1, 11))
        )

        ax0.plot(
            x.index,
            100.0 * x.values,
            marker=markers[tracker],
            linestyle=linestyles[storm],
            linewidth=2.0,
            markersize=6,
            label=f"{storm} — {tracker.capitalize()}",
        )


ax0.axvline(
    9.5,
    linestyle=":",
    linewidth=2,
)

ax0.axvspan(
    9.5,
    10.5,
    alpha=0.08,
)

ax0.text(
    9.58,
    96,
    "Cycle 10\ntermination",
    ha="left",
    va="top",
    fontsize=10,
)

ax0.set_xlim(0.7, 10.3)
ax0.set_ylim(-4, 104)
ax0.set_xticks(range(1, 11))
ax0.set_yticks([0, 25, 50, 75, 100])

ax0.set_xlabel("Operational cycle")
ax0.set_ylabel("Detection across four models (%)")

ax0.set_title(
    "A. Operational tracker detection evolution"
)

ax0.grid(alpha=0.25)

ax0.legend(
    ncol=3,
    fontsize=9,
    loc="lower left",
)


# ---------------------------------------------------------------------
# Variability plotting helper
# ---------------------------------------------------------------------

def plot_variability(ax, storm, panel):

    df = var[storm].copy()

    valid = df[df["comparison_valid"]].copy()

    ax.plot(
        df["valid_time"],
        df["historical_r67_km"],
        linestyle="-",
        linewidth=2.0,
        label="Historical r67",
    )

    ax.plot(
        df["valid_time"],
        df["historical_r90_km"],
        linestyle="-",
        linewidth=2.0,
        label="Historical r90",
    )

    # Plot recent-3 spread only where the comparison is valid.
    # This prevents unsupported terminal values from appearing as
    # physical convergence toward zero spread.
    recent_r67 = df["recent_r67_km"].where(
        df["comparison_valid"]
    )
    recent_r90 = df["recent_r90_km"].where(
        df["comparison_valid"]
    )

    ax.plot(
        df["valid_time"],
        recent_r67,
        linestyle="--",
        linewidth=2.0,
        label="Recent-3 r67",
    )

    ax.plot(
        df["valid_time"],
        recent_r90,
        linestyle="--",
        linewidth=2.0,
        label="Recent-3 r90",
    )

    # Highlight only statistically accepted comparison interval.
    if not valid.empty:

        ax.scatter(
            valid["valid_time"],
            valid["recent_r67_km"],
            s=35,
            zorder=5,
        )

        ax.scatter(
            valid["valid_time"],
            valid["recent_r90_km"],
            s=35,
            zorder=5,
        )

        mean67 = valid[
            "r67_spread_contraction_pct"
        ].mean()

        mean90 = valid[
            "r90_spread_contraction_pct"
        ].mean()

        med67 = valid[
            "r67_spread_contraction_pct"
        ].median()

        med90 = valid[
            "r90_spread_contraction_pct"
        ].median()

        text = (
            "Valid recent-3 comparison\n"
            f"Mean contraction: "
            f"r67={mean67:.1f}%, "
            f"r90={mean90:.1f}%\n"
            f"Median contraction: "
            f"r67={med67:.1f}%, "
            f"r90={med90:.1f}%"
        )

        ax.text(
            0.015,
            0.96,
            text,
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=9,
            bbox=dict(
                boxstyle="round,pad=0.35",
                facecolor="white",
                alpha=0.85,
            ),
        )

    ax.set_ylabel("Track spread radius (km)")

    ax.set_title(
        f"{panel}. {storm} — WuDuan track convergence"
    )

    ax.grid(alpha=0.25)

    ax.legend(
        ncol=4,
        fontsize=9,
        loc="upper center",
        bbox_to_anchor=(0.60, 1.0),
    )


plot_variability(
    ax1,
    "Odalys",
    "B",
)

plot_variability(
    ax2,
    "Polo",
    "C",
)

ax2.set_xlabel("Valid time (UTC)")

for ax in (ax1, ax2):
    ax.tick_params(
        axis="x",
        rotation=25,
    )


# ---------------------------------------------------------------------
# Main title / experiment definition
# ---------------------------------------------------------------------

fig.suptitle(
    "AIWeather Odalys–Polo Operational Experiment\n"
    "Cycles 1–9 accepted | Cycle 10 operational termination",
    fontsize=16,
    y=0.985,
)


# ---------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------

png = OUT / "odalys_polo_operational_experiment_summary.png"
pdf = OUT / "odalys_polo_operational_experiment_summary.pdf"

fig.savefig(
    png,
    dpi=220,
    bbox_inches="tight",
)

fig.savefig(
    pdf,
    bbox_inches="tight",
)

plt.close(fig)


# ---------------------------------------------------------------------
# Numerical summary
# ---------------------------------------------------------------------

print("=" * 78)
print("ODALYS–POLO FINAL OPERATIONAL EXPERIMENT SUMMARY")
print("=" * 78)

print()
print("Detection population:")
print("  Cycles 1–9 : accepted")
print("  Cycle 10   : termination / excluded from variability")

cycle10 = det[det["cycle"] == 10]

print()
print(
    "Cycle 10 detections:",
    int(cycle10["detected"].sum()),
    "/",
    len(cycle10),
)

for storm in storms:

    valid = var[storm][
        var[storm]["comparison_valid"]
    ]

    print()
    print(storm.upper())
    print("-" * 40)

    print(
        "Valid comparison times:",
        len(valid),
    )

    if not valid.empty:

        for radius in ("r67", "r90"):

            values = valid[
                f"{radius}_spread_contraction_pct"
            ]

            print(
                f"{radius} contraction: "
                f"mean={values.mean():.1f}%  "
                f"median={values.median():.1f}%  "
                f"range={values.min():.1f}"
                f"–{values.max():.1f}%"
            )


print()
print("Saved:")
print(png)
print(pdf)
print()
print("DONE")
