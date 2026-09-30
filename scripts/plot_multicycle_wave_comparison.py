"""
Compare AIFS2 storm-relative wave forecasts between operational cycles.

The script reads existing wave diagnostics only. It does not rerun
forecast inference, tropical-cyclone tracking, or wave analysis.

Forecast cycles are compared at common valid times rather than common
forecast lead times.
"""

import argparse
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd


DEFAULT_STORMS = ("Odalys", "Polo")
WAVE_ROOT = Path("results/waves")
INIT_HOUR_UTC = 12


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Compare consecutive operational AIFS2 wave "
            "forecast cycles for one or more tropical cyclones."
        )
    )

    parser.add_argument(
        "--storms",
        nargs="+",
        default=list(DEFAULT_STORMS),
        help=(
            "Storms to compare. Default: Odalys Polo."
        ),
    )

    return parser.parse_args()


ARGS = parse_args()
STORMS = tuple(ARGS.storms)


def discover_cases():
    """
    Discover completed operational AIFS2 wave cycles.

    Only directories following

        aifs2_<storm>_YYYYMMDD

    are considered. Historical undated wave cases are ignored.

    A cycle is retained only when diagnostics exist for every
    storm in STORMS.
    """
    discovered = {
        storm: {}
        for storm in STORMS
    }

    for storm in STORMS:
        storm_lower = storm.lower()

        pattern = (
            f"aifs2_{storm_lower}_"
            "[0-9][0-9][0-9][0-9]"
            "[0-9][0-9][0-9][0-9]"
        )

        for directory in sorted(
            WAVE_ROOT.glob(pattern)
        ):
            date_string = directory.name.rsplit(
                "_",
                1,
            )[-1]

            try:
                date = pd.to_datetime(
                    date_string,
                    format="%Y%m%d",
                )
            except ValueError:
                continue

            diagnostics = directory / (
                f"aifs2_{storm_lower}_"
                "wave_diagnostics.csv"
            )

            if not diagnostics.is_file():
                continue

            init = date + pd.Timedelta(
                hours=INIT_HOUR_UTC
            )

            cycle_key = init.strftime(
                "%Y%m%dT%H%M%S"
            )

            discovered[storm][cycle_key] = {
                "init": init.strftime(
                    "%Y-%m-%d %H:%M:%S"
                ),
                "path": diagnostics,
            }

    common_cycles = set.intersection(
        *(
            set(discovered[storm])
            for storm in STORMS
        )
    )

    common_cycles = sorted(common_cycles)

    if len(common_cycles) < 2:
        raise RuntimeError(
            "At least two complete common operational "
            "wave cycles are required."
        )

    cases = {
        storm: {}
        for storm in STORMS
    }

    for cycle in common_cycles:
        timestamp = pd.to_datetime(
            cycle,
            format="%Y%m%dT%H%M%S",
        )

        label = timestamp.strftime(
            "%Y-%m-%d %H UTC"
        )

        for storm in STORMS:
            cases[storm][label] = (
                discovered[storm][cycle]
            )

    return cases, common_cycles


CASES, CYCLE_KEYS = discover_cases()

# Labels are already chronological because CYCLE_KEYS is sorted
# and discover_cases() inserts cases in that order.
CYCLE_LABELS = list(
    CASES[STORMS[0]]
)

# Build consecutive forecast-cycle comparisons automatically:
# cycle 2 - cycle 1, cycle 3 - cycle 2, etc.
revision_pairs = []

for index in range(1, len(CYCLE_LABELS)):
    old_label = CYCLE_LABELS[index - 1]
    new_label = CYCLE_LABELS[index]

    old_time = pd.to_datetime(
        CYCLE_KEYS[index - 1],
        format="%Y%m%dT%H%M%S",
    )
    new_time = pd.to_datetime(
        CYCLE_KEYS[index],
        format="%Y%m%dT%H%M%S",
    )

    revision_label = (
        f"{new_time.strftime('%d %b')} - "
        f"{old_time.strftime('%d %b')}"
    )

    revision_pairs.append(
        (
            old_label,
            new_label,
            revision_label,
        )
    )


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


n_storms = len(STORMS)

fig, axes = plt.subplots(
    3,
    n_storms,
    figsize=(7 * n_storms, 11),
    sharex="col",
    constrained_layout=True,
    squeeze=False,
)

line_styles = [
    "-",
    "--",
    ":",
    "-.",
]

cycle_styles = {
    label: line_styles[
        index % len(line_styles)
    ]
    for index, label in enumerate(
        CASES[STORMS[0]]
    )
}


for column, storm in enumerate(STORMS):

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
    for revision_index, (
        old_label,
        new_label,
        revision_label,
    ) in enumerate(revision_pairs):

        revision_style = line_styles[
            revision_index % len(line_styles)
        ]

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


first_cycle = pd.to_datetime(
    CYCLE_KEYS[0],
    format="%Y%m%dT%H%M%S",
)

last_cycle = pd.to_datetime(
    CYCLE_KEYS[-1],
    format="%Y%m%dT%H%M%S",
)

if (
    first_cycle.year == last_cycle.year
    and first_cycle.month == last_cycle.month
):
    cycle_range_title = (
        f"{first_cycle.strftime('%d')}–"
        f"{last_cycle.strftime('%d %B %Y')}"
    )
else:
    cycle_range_title = (
        f"{first_cycle.strftime('%d %b %Y')} – "
        f"{last_cycle.strftime('%d %b %Y')}"
    )

fig.suptitle(
    "AIFS2 Operational Wave Forecast Cycle Comparison\n"
    f"{' and '.join(STORMS)} — {cycle_range_title}",
    fontsize=14,
)


storm_slug = "_".join(
    storm.lower()
    for storm in STORMS
)

output = Path(
    "outputs/verification/"
    f"{storm_slug}_wave_cycles_"
    f"{first_cycle.strftime('%Y%m%d')}_"
    f"{last_cycle.strftime('%Y%m%d')}.png"
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

for storm in STORMS:

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
