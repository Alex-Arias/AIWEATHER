"""
Compare AIFS2 storm-relative wave forecasts between operational cycles.

The script reads existing wave diagnostics only. It does not rerun
forecast inference, tropical-cyclone tracking, or wave analysis.

Forecast cycles are compared at common valid times rather than common
forecast lead times.
"""

from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd


CASES = {
    "Odalys": {
        "2026-09-20 12 UTC": {
            "init": "2026-09-20 12:00:00",
            "path": Path(
                "results/waves/aifs2_odalys_20260920/"
                "aifs2_odalys_wave_diagnostics.csv"
            ),
        },
        "2026-09-21 12 UTC": {
            "init": "2026-09-21 12:00:00",
            "path": Path(
                "results/waves/aifs2_odalys_20260921/"
                "aifs2_odalys_wave_diagnostics.csv"
            ),
        },
    },
    "Polo": {
        "2026-09-20 12 UTC": {
            "init": "2026-09-20 12:00:00",
            "path": Path(
                "results/waves/aifs2_polo_20260920/"
                "aifs2_polo_wave_diagnostics.csv"
            ),
        },
        "2026-09-21 12 UTC": {
            "init": "2026-09-21 12:00:00",
            "path": Path(
                "results/waves/aifs2_polo_20260921/"
                "aifs2_polo_wave_diagnostics.csv"
            ),
        },
    },
}


def load_case(config):
    """Read diagnostics and reconstruct forecast valid time."""

    df = pd.read_csv(config["path"])

    init = pd.Timestamp(
        config["init"],
        tz="UTC",
    )

    df["valid_time"] = (
        init
        + pd.to_timedelta(
            df["lead_time_hours"],
            unit="h",
        )
    )

    return df


data = {}

for storm, cycles in CASES.items():
    data[storm] = {}

    for label, config in cycles.items():
        data[storm][label] = load_case(config)


fig, axes = plt.subplots(
    3,
    2,
    figsize=(14, 11),
    sharex="col",
    constrained_layout=True,
)

cycle_styles = {
    "2026-09-20 12 UTC": "-",
    "2026-09-21 12 UTC": "--",
}


for column, storm in enumerate(("Odalys", "Polo")):

    ax_swh = axes[0, column]
    ax_wind = axes[1, column]
    ax_delta = axes[2, column]

    for label, df in data[storm].items():

        line, = ax_swh.plot(
            df["valid_time"],
            df["max_swh_300km_m"],
            linestyle=cycle_styles[label],
            linewidth=2.0,
            marker="o",
            markersize=3,
            label=label,
        )

        ax_wind.plot(
            df["valid_time"],
            df["max_wind_300km_ms"],
            linestyle=cycle_styles[label],
            linewidth=2.0,
            marker="o",
            markersize=3,
            color=line.get_color(),
            label=label,
        )

    # Compare the two cycles at identical valid times.
    old = data[storm]["2026-09-20 12 UTC"]
    new = data[storm]["2026-09-21 12 UTC"]

    common = old[
        [
            "valid_time",
            "max_swh_300km_m",
        ]
    ].merge(
        new[
            [
                "valid_time",
                "max_swh_300km_m",
            ]
        ],
        on="valid_time",
        suffixes=("_20", "_21"),
    )

    common["delta_swh_m"] = (
        common["max_swh_300km_m_21"]
        - common["max_swh_300km_m_20"]
    )

    ax_delta.plot(
        common["valid_time"],
        common["delta_swh_m"],
        linewidth=2.0,
        marker="o",
        markersize=3,
    )

    ax_delta.axhline(
        0.0,
        linewidth=1.0,
        linestyle="--",
    )

    # Mark the largest absolute cycle-to-cycle SWH revision.
    if not common.empty:
        peak_index = (
            common["delta_swh_m"]
            .abs()
            .idxmax()
        )
        peak = common.loc[peak_index]

        ax_delta.scatter(
            peak["valid_time"],
            peak["delta_swh_m"],
            s=55,
            zorder=5,
        )

        ax_delta.annotate(
            f"{peak['delta_swh_m']:+.2f} m",
            xy=(
                peak["valid_time"],
                peak["delta_swh_m"],
            ),
            xytext=(8, 8),
            textcoords="offset points",
            fontsize=9,
        )

    ax_swh.set_title(storm)

    ax_swh.set_ylabel(
        "Maximum SWH within 300 km (m)"
    )

    ax_wind.set_ylabel(
        "Maximum wind within 300 km (m s$^{-1}$)"
    )

    ax_delta.set_ylabel(
        "$\\Delta$SWH (21 Sep - 20 Sep) (m)"
    )

    ax_delta.set_xlabel(
        "Forecast valid time (UTC)"
    )

    for ax in (ax_swh, ax_wind, ax_delta):
        ax.grid(
            True,
            alpha=0.3,
        )

        ax.xaxis.set_major_locator(
            mdates.DayLocator(interval=1)
        )

        ax.xaxis.set_major_formatter(
            mdates.DateFormatter("%d Sep")
        )

    ax_swh.legend(
        title="Initialization",
    )


fig.suptitle(
    "AIFS2 Operational Wave Forecast Cycle Comparison\n"
    "Odalys and Polo — 20 vs 21 September 2026",
    fontsize=14,
)


output = Path(
    "outputs/verification/"
    "polo_odalys_wave_cycles_"
    "20260920_20260921.png"
)

output.parent.mkdir(
    parents=True,
    exist_ok=True,
)

fig.savefig(
    output,
    dpi=180,
    bbox_inches="tight",
)

print(output)


# ------------------------------------------------------------
# Common-valid-time numerical comparison
# ------------------------------------------------------------

print()
print("COMMON VALID-TIME COMPARISON")
print("=" * 72)

for storm in ("Odalys", "Polo"):

    old = data[storm]["2026-09-20 12 UTC"]
    new = data[storm]["2026-09-21 12 UTC"]

    merged = old[
        [
            "valid_time",
            "max_swh_300km_m",
            "max_wind_300km_ms",
        ]
    ].merge(
        new[
            [
                "valid_time",
                "max_swh_300km_m",
                "max_wind_300km_ms",
            ]
        ],
        on="valid_time",
        suffixes=("_20", "_21"),
    )

    merged["delta_swh_m"] = (
        merged["max_swh_300km_m_21"]
        - merged["max_swh_300km_m_20"]
    )

    merged["delta_wind_ms"] = (
        merged["max_wind_300km_ms_21"]
        - merged["max_wind_300km_ms_20"]
    )

    print(f"\n{storm}")
    print(
        f"common times : {len(merged)}"
    )

    if len(merged):
        print(
            "mean ΔSWH   : "
            f"{merged['delta_swh_m'].mean():+.2f} m"
        )
        print(
            "max |ΔSWH| : "
            f"{merged['delta_swh_m'].abs().max():.2f} m"
        )
        print(
            "mean Δwind  : "
            f"{merged['delta_wind_ms'].mean():+.2f} m/s"
        )
