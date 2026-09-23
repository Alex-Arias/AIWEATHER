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
        "2026-09-22 12 UTC": {
            "init": "2026-09-22 12:00:00",
            "path": Path(
                "results/waves/aifs2_odalys_20260922/"
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
        "2026-09-22 12 UTC": {
            "init": "2026-09-22 12:00:00",
            "path": Path(
                "results/waves/aifs2_polo_20260922/"
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
    "2026-09-22 12 UTC": ":",
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

    # Compare successive operational cycles at identical
    # forecast valid times. This shows how the storm-relative
    # SWH forecast changes as initialization advances.
    revision_pairs = [
        (
            "2026-09-20 12 UTC",
            "2026-09-21 12 UTC",
            "21 Sep - 20 Sep",
            "-",
        ),
        (
            "2026-09-21 12 UTC",
            "2026-09-22 12 UTC",
            "22 Sep - 21 Sep",
            "--",
        ),
    ]

    for (
        old_label,
        new_label,
        revision_label,
        revision_style,
    ) in revision_pairs:

        old = data[storm][old_label]
        new = data[storm][new_label]

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
            suffixes=("_old", "_new"),
        )

        common["delta_swh_m"] = (
            common["max_swh_300km_m_new"]
            - common["max_swh_300km_m_old"]
        )

        revision_line, = ax_delta.plot(
            common["valid_time"],
            common["delta_swh_m"],
            linewidth=2.0,
            linestyle=revision_style,
            marker="o",
            markersize=3,
            label=revision_label,
        )

        # Mark the largest absolute SWH revision for each
        # consecutive cycle pair.
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
                color=revision_line.get_color(),
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
                color=revision_line.get_color(),
            )

    ax_delta.axhline(
        0.0,
        linewidth=1.0,
        linestyle=":",
        color="black",
    )

    ax_swh.set_title(storm)

    ax_swh.set_ylabel(
        "Maximum SWH within 300 km (m)"
    )

    ax_wind.set_ylabel(
        "Maximum wind within 300 km (m s$^{-1}$)"
    )

    ax_delta.set_ylabel(
        "$\\Delta$SWH between successive cycles (m)"
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
            mdates.DateFormatter("%d %b")
        )

    ax_swh.legend(
        title="Initialization",
    )

    ax_delta.legend(
        title="Cycle revision",
        fontsize=9,
    )


fig.suptitle(
    "AIFS2 Operational Wave Forecast Cycle Comparison\n"
    "Odalys and Polo — 20–22 September 2026",
    fontsize=14,
)


output = Path(
    "outputs/verification/"
    "polo_odalys_wave_cycles_"
    "20260920_20260922.png"
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

revision_pairs = [
    (
        "2026-09-20 12 UTC",
        "2026-09-21 12 UTC",
        "21 Sep - 20 Sep",
    ),
    (
        "2026-09-21 12 UTC",
        "2026-09-22 12 UTC",
        "22 Sep - 21 Sep",
    ),
]

for storm in ("Odalys", "Polo"):

    print()
    print(storm.upper())
    print("-" * 72)

    for old_label, new_label, label in revision_pairs:

        old = data[storm][old_label]
        new = data[storm][new_label]

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
            suffixes=("_old", "_new"),
        )

        merged["delta_swh_m"] = (
            merged["max_swh_300km_m_new"]
            - merged["max_swh_300km_m_old"]
        )

        merged["delta_wind_ms"] = (
            merged["max_wind_300km_ms_new"]
            - merged["max_wind_300km_ms_old"]
        )

        print()
        print(label)
        print(
            f"common times : {len(merged)}"
        )

        if merged.empty:
            continue

        print(
            "mean dSWH   : "
            f"{merged['delta_swh_m'].mean():+.3f} m"
        )

        print(
            "max |dSWH| : "
            f"{merged['delta_swh_m'].abs().max():.3f} m"
        )

        print(
            "mean dWind  : "
            f"{merged['delta_wind_ms'].mean():+.3f} m/s"
        )

        print(
            "max |dWind|: "
            f"{merged['delta_wind_ms'].abs().max():.3f} m/s"
        )
