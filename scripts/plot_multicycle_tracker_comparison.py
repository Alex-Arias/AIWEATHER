"""
Plot multicycle operational tropical-cyclone tracks by tracker.

Rows represent storms and columns represent tracking systems:
Native, WuDuan, and Vitart.

Color identifies the forecast model.
Line style identifies the forecast initialization/cycle.

The script reads existing operational track CSV files only. It does
not rerun a tracker and does not use IBTrACS.
"""

import argparse
from datetime import datetime, timedelta
from pathlib import Path

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
from matplotlib.lines import Line2D
import pandas as pd

from aiweather.plotting.tracks import normalize_longitude
try:
    from scripts.tc_translation_speed import calculate_translation_speed
except ModuleNotFoundError as exc:
    if exc.name != "scripts":
        raise
    from tc_translation_speed import calculate_translation_speed


TRACKERS = [
    "native",
    "wuduan",
    "vitart",
]

MODELS = [
    "graphcast",
    "aifs2",
    "pangu3",
    "pangu6",
]

MODEL_COLORS = {
    "graphcast": "C0",
    "aifs2": "C1",
    "pangu3": "C2",
    "pangu6": "C3",
}

MODEL_LABELS = {
    "graphcast": "GraphCast",
    "aifs2": "AIFS2",
    "pangu3": "Pangu3",
    "pangu6": "Pangu6",
}

TRACKER_LABELS = {
    "native": "Native",
    "wuduan": "WuDuan",
    "vitart": "Vitart",
}

DOMAINS = {
    "Odalys": (-140.0, -110.0, 8.0, 34.0),
    "Polo": (-125.0, -90.0, 8.0, 34.0),
    "Rachel": (-125.0, -90.0, 5.0, 30.0),
    "INVEST 92E": (-120.0, -90.0, 5.0, 35.0),
    "Simon": (-120.0, -90.0, 5.0, 35.0),
}



DATE_LABEL_OFFSETS = {
    "graphcast": (5, 7),
    "aifs2": (5, -12),
    "pangu3": (-32, 7),
    "pangu6": (-32, -12),
}


FOCUSED_DOMAINS = {
    "Simon": (-110.0, -99.0, 13.0, 26.0),
    "INVEST 92E": (-110.0, -99.0, 13.0, 26.0),
}


# Forecast-cycle line styles.
#
# The first four preserve the original Cycle 1–4 appearance.
# Additional dash patterns allow later operational cycles to be
# added without changing the plotting code.
CYCLE_LINESTYLES = [
    "-",
    "--",
    "-.",
    ":",
    (0, (5, 1)),
    (0, (3, 1, 1, 1)),
    (0, (1, 1)),
    (0, (5, 2, 1, 2)),
]


def cycle_linestyle(cycle_index):
    """Return a reusable line style for a zero-based cycle index."""
    if cycle_index < 0:
        raise ValueError("cycle_index must be non-negative.")

    return CYCLE_LINESTYLES[
        cycle_index % len(CYCLE_LINESTYLES)
    ]


def track_path(
    storm,
    model,
    tracker,
    init,
    experiment_slug=None,
):
    slug = (
        storm.lower().replace(" ", "_")
        if experiment_slug is None
        else experiment_slug
    )

    base = (
        Path("results/operational")
        / f"{slug}_{init}"
        / model
    )

    if tracker == "native":
        filename = f"{model}_track.csv"
    else:
        filename = f"{model}_{tracker}_track.csv"

    return base / filename


def load_track(
    storm,
    model,
    tracker,
    init,
    experiment_slug=None,
):
    path = track_path(
        storm,
        model,
        tracker,
        init,
        experiment_slug=experiment_slug,
    )

    if not path.exists():
        return None

    df = pd.read_csv(path)

    if df.empty:
        return None

    required = {
        "lead_time_hours",
        "latitude",
        "longitude",
    }

    missing = required.difference(df.columns)

    if missing:
        raise ValueError(
            f"{path} is missing columns: "
            f"{sorted(missing)}"
        )

    df = df.copy()

    df["longitude_plot"] = (
        df["longitude"]
        .astype(float)
        .map(normalize_longitude)
    )

    df["latitude"] = df["latitude"].astype(float)
    df["lead_time_hours"] = (
        df["lead_time_hours"].astype(float)
    )

    return df



def prepare_translation_speed(track, init_time):
    """
    Calculate cyclone translation speed and absolute UTC times.

    Parameters
    ----------
    track : pandas.DataFrame
        Existing operational tracker positions.

    init_time : datetime
        Forecast initialization in UTC.

    Returns
    -------
    pandas.DataFrame
        Track positions with translation_speed_kmh and valid_time.

    Notes
    -----
    Translation speed represents horizontal displacement
    between successive forecast positions divided by the
    actual elapsed time.

    It is not maximum sustained wind speed.
    """
    result = calculate_translation_speed(track)

    result["valid_time"] = (
        pd.Timestamp(init_time)
        + pd.to_timedelta(
            result["lead_time_hours"],
            unit="h",
        )
    )

    return result


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Plot multiple operational tropical-cyclone forecast "
            "cycles for multiple trackers and forecast models."
        )
    )

    parser.add_argument(
        "--inits",
        nargs="+",
        required=True,
        help=(
            "Forecast initializations in YYYYMMDDTHHMMSS format, "
            "ordered from earliest to latest."
        ),
    )

    parser.add_argument(
        "--storms",
        nargs="+",
        required=True,
        help="Storm names to plot, one row per storm.",
    )

    parser.add_argument(
        "--output",
        default=None,
        help="Optional output PNG path.",
    )

    parser.add_argument(
        "--experiment-slug",
        default=None,
        help=(
            "Persistent operational experiment directory slug. "
            "Defaults to the normalized storm name."
        ),
    )

    parser.add_argument(
        "--show-utc-dates",
        action="store_true",
        help="Annotate track positions with UTC dates.",
    )

    parser.add_argument(
        "--show-translation-speed",
        action="store_true",
        help="Add translation-speed panels using absolute UTC time.",
    )

    parser.add_argument(
        "--focused-domain",
        action="store_true",
        help="Use a focused geographic domain when available.",
    )

    args = parser.parse_args()

    if args.experiment_slug is not None:
        slug = args.experiment_slug

        if not slug or not all(
            character.islower()
            or character.isdigit()
            or character == "_"
            for character in slug
        ):
            raise ValueError(
                "--experiment-slug must contain only lowercase "
                "letters, digits, and underscores."
            )

    init_times = []

    for init in args.inits:
        try:
            init_times.append(
                datetime.strptime(
                    init,
                    "%Y%m%dT%H%M%S",
                )
            )
        except ValueError as exc:
            raise ValueError(
                "--inits must use YYYYMMDDTHHMMSS format"
            ) from exc

    storms = args.storms

    unknown = [
        storm
        for storm in storms
        if storm not in DOMAINS
    ]

    if unknown:
        raise ValueError(
            "No plotting domain configured for: "
            + ", ".join(unknown)
        )

    projection = ccrs.PlateCarree()

    if args.show_translation_speed:
        fig = plt.figure(figsize=(18, 11))

        grid = fig.add_gridspec(
            nrows=2 * len(storms),
            ncols=len(TRACKERS),
            height_ratios=[
                value
                for _ in storms
                for value in (2.6, 1.2)
            ],
            hspace=0.22,
            wspace=0.12,
        )

        axes = np.empty(
            (len(storms), len(TRACKERS)),
            dtype=object,
        )

        speed_axes = np.empty(
            (len(storms), len(TRACKERS)),
            dtype=object,
        )

        for row in range(len(storms)):
            for col in range(len(TRACKERS)):
                axes[row, col] = fig.add_subplot(
                    grid[2 * row, col],
                    projection=projection,
                )

                speed_axes[row, col] = fig.add_subplot(
                    grid[2 * row + 1, col],
                )

    else:
        fig, axes = plt.subplots(
            len(storms),
            len(TRACKERS),
            figsize=(17, 10),
            subplot_kw={
                "projection": projection,
            },
            squeeze=False,
        )

        speed_axes = None

    loaded_track_count = 0

    speed_times = []
    speed_values = []

    for row, storm in enumerate(storms):
        (
            lon_min,
            lon_max,
            lat_min,
            lat_max,
        ) = (
            FOCUSED_DOMAINS.get(storm, DOMAINS[storm])
            if args.focused_domain
            else DOMAINS[storm]
        )

        for col, tracker in enumerate(TRACKERS):
            ax = axes[row, col]

            ax.set_extent(
                [
                    lon_min,
                    lon_max,
                    lat_min,
                    lat_max,
                ],
                crs=projection,
            )

            ax.add_feature(
                cfeature.LAND,
                facecolor="0.92",
                zorder=0,
            )
            ax.add_feature(
                cfeature.OCEAN,
                facecolor="white",
                zorder=0,
            )
            ax.add_feature(
                cfeature.COASTLINE,
                linewidth=0.8,
                zorder=2,
            )
            ax.add_feature(
                cfeature.BORDERS,
                linewidth=0.5,
                zorder=2,
            )

            gridlines = ax.gridlines(
                crs=projection,
                draw_labels=True,
                linewidth=0.5,
                alpha=0.35,
                linestyle="--",
            )

            if args.focused_domain:
                from matplotlib.ticker import FixedLocator

                gridlines.xlocator = FixedLocator(
                    np.arange(
                        np.ceil(lon_min / 2.0) * 2.0,
                        lon_max + 0.01,
                        2.0,
                    )
                )

            gridlines.top_labels = False
            gridlines.right_labels = False

            if col > 0:
                gridlines.left_labels = False

            # -------------------------------------------------
            # Tracks:
            # color = model
            # linestyle = cycle
            # -------------------------------------------------

            for cycle_index, init in enumerate(args.inits):
                linestyle = cycle_linestyle(cycle_index)

                for model in MODELS:
                    df = load_track(
                        storm,
                        model,
                        tracker,
                        init,
                        experiment_slug=args.experiment_slug,
                    )

                    if df is None:
                        continue

                    loaded_track_count += 1
                    color = MODEL_COLORS[model]

                    ax.plot(
                        df["longitude_plot"],
                        df["latitude"],
                        linewidth=1.8,
                        linestyle=linestyle,
                        color=color,
                        alpha=0.90,
                        transform=projection,
                        zorder=3,
                    )

                    if args.show_translation_speed:
                        speed_df = prepare_translation_speed(
                            df,
                            init_times[cycle_index],
                        )

                        valid_speed = speed_df[
                            "translation_speed_kmh"
                        ].notna()

                        if valid_speed.any():
                            speed_times.extend(
                                speed_df.loc[
                                    valid_speed, "valid_time"
                                ].tolist()
                            )

                            speed_values.extend(
                                speed_df.loc[
                                    valid_speed,
                                    "translation_speed_kmh",
                                ].tolist()
                            )

                        speed_ax = speed_axes[row, col]

                        speed_ax.plot(
                            speed_df["valid_time"],
                            speed_df["translation_speed_kmh"],
                            color=color,
                            linestyle=linestyle,
                            linewidth=1.5,
                            alpha=0.85,
                        )

                        if tracker == "native":
                            flagged = speed_df.loc[
                                speed_df["translation_speed_kmh"] > 30.0
                            ]

                            if not flagged.empty:
                                speed_ax.scatter(
                                    flagged["valid_time"],
                                    flagged["translation_speed_kmh"],
                                    facecolors="none",
                                    edgecolors=color,
                                    s=45,
                                    linewidths=1.2,
                                    zorder=6,
                                )

                    # First available tracker point.
                    first = df.iloc[0]

                    ax.scatter(
                        first["longitude_plot"],
                        first["latitude"],
                        marker="o",
                        s=34,
                        color=color,
                        edgecolor="white",
                        linewidth=0.55,
                        alpha=0.90,
                        transform=projection,
                        zorder=5,
                    )

                    # Common 24-hour forecast markers.
                    lead = (
                        df["lead_time_hours"]
                        .round()
                        .astype(int)
                    )

                    marker_mask = (
                        (lead > 0)
                        & (lead % 24 == 0)
                    )

                    marker_df = df.loc[marker_mask]

                    if not marker_df.empty:
                        ax.scatter(
                            marker_df["longitude_plot"],
                            marker_df["latitude"],
                            marker="o",
                            s=13,
                            color=color,
                            edgecolor="white",
                            linewidth=0.35,
                            alpha=0.85,
                            transform=projection,
                            zorder=4,
                        )

                    # UTC calendar dates for the latest cycle only.
                    # Existing 24-hour markers remain unchanged.
                    if (
                        args.show_utc_dates
                        and cycle_index == len(args.inits) - 1
                        and model == "aifs2"
                    ):
                        init_time = init_times[cycle_index]

                        for _, point in df.iterrows():
                            valid_time = (
                                init_time
                                + timedelta(
                                    hours=float(point["lead_time_hours"])
                                )
                            )

                            if (
                                valid_time.hour != 0
                                or valid_time.minute != 0
                            ):
                                continue

                            dx, dy = DATE_LABEL_OFFSETS[model]

                            ax.annotate(
                                valid_time.strftime("%d %b"),
                                xy=(
                                    point["longitude_plot"],
                                    point["latitude"],
                                ),
                                xycoords=projection._as_mpl_transform(ax),
                                xytext=(dx, dy),
                                textcoords="offset points",
                                fontsize=7,
                                color=color,
                                alpha=0.9,
                                annotation_clip=True,
                                zorder=6,
                            )

            if row == 0:
                ax.set_title(
                    f"{TRACKER_LABELS[tracker]} tracker",
                    fontsize=14,
                    fontweight="bold",
                    y=1.02,
                    pad=4,
                )

            if col == 0:
                ax.text(
                    -0.15,
                    0.5,
                    storm,
                    transform=ax.transAxes,
                    rotation=90,
                    va="center",
                    ha="center",
                    fontsize=14,
                    fontweight="bold",
                )

    if args.show_translation_speed:
        for row in range(len(storms)):
            for col, tracker in enumerate(TRACKERS):
                speed_ax = speed_axes[row, col]

                speed_ax.set_title(
                    f"{TRACKER_LABELS[tracker]} translation speed",
                    fontsize=10,
                )

                speed_ax.set_ylabel(
                    "Speed (km/h)",
                    fontsize=9,
                )

                speed_ax.set_xlabel(
                    "Valid time (UTC)",
                    fontsize=9,
                )

                speed_ax.grid(
                    True,
                    alpha=0.3,
                    linestyle="--",
                )

                speed_ax.xaxis.set_major_locator(
                    mdates.DayLocator(interval=1)
                )

                speed_ax.xaxis.set_major_formatter(
                    mdates.DateFormatter("%d %b")
                )

                speed_ax.tick_params(
                    axis="x",
                    labelrotation=35,
                    labelsize=8,
                )

                speed_ax.tick_params(
                    axis="y",
                    labelsize=8,
                )

                speed_ax.set_ylim(bottom=0)

    if args.show_translation_speed and speed_times:
        time_min = min(speed_times)
        time_max = max(speed_times)

        speed_max = max(speed_values)

        common_speed_max = max(
            10.0,
            speed_max * 1.08,
        )

        for speed_ax in speed_axes.flat:
            speed_ax.set_xlim(
                time_min,
                time_max,
            )

            speed_ax.set_ylim(
                0,
                common_speed_max,
            )

    # ---------------------------------------------------------
    # Model legend: color
    # ---------------------------------------------------------

    model_handles = [
        Line2D(
            [0],
            [0],
            color=MODEL_COLORS[model],
            linewidth=2.2,
            label=MODEL_LABELS[model],
        )
        for model in MODELS
    ]

    model_legend = fig.legend(
        handles=model_handles,
        title="Forecast model",
        loc="lower center",
        ncol=4,
        frameon=True,
        bbox_to_anchor=(0.5, 0.075),
    )

    # Keep first legend when adding second one.
    fig.add_artist(model_legend)

    # ---------------------------------------------------------
    # Cycle legend: line style
    # ---------------------------------------------------------

    cycle_handles = []

    for cycle_index, init_time in enumerate(init_times):
        cycle_handles.append(
            Line2D(
                [0],
                [0],
                color="black",
                linewidth=2.0,
                linestyle=cycle_linestyle(cycle_index),
                label=(
                    f"Cycle {cycle_index + 1}: "
                    f"{init_time.strftime('%d %b %H UTC')}"
                ),
            )
        )

    fig.legend(
        handles=cycle_handles,
        title="Forecast cycle",
        loc="lower center",
        ncol=len(cycle_handles),
        frameon=True,
        bbox_to_anchor=(0.5, 0.025),
    )

    first_time = init_times[0]
    last_time = init_times[-1]

    fig.suptitle(
        "AIWeather Multicycle Operational Track Comparison\n"
        f"{first_time.strftime('%d %b %Y %H UTC')} – "
        f"{last_time.strftime('%d %b %Y %H UTC')} "
        "— forecast-only tracks",
        fontsize=16,
        y=0.975,
    )

    fig.text(
        0.5,
        0.002,
        (
            "Color = forecast model; line style = forecast cycle; "
            "track markers every 24 h. "
            "Speed panels use absolute UTC valid time. "
            "Open circles: Native speed >30 km/h "
            "(diagnostic only)."
            if args.show_translation_speed
            else
            "Color = forecast model; line style = forecast cycle; "
            "markers every 24 h."
        ),
        ha="center",
        va="bottom",
        fontsize=9,
    )

    if args.show_translation_speed:
        fig.subplots_adjust(
            left=0.07,
            right=0.985,
            bottom=0.19,
            top=0.88,
        )
    else:
        fig.subplots_adjust(
            left=0.07,
            right=0.985,
            bottom=0.19,
            top=0.88,
            wspace=0.08,
            hspace=0.16,
        )

    if args.output is None:
        output = (
            Path("results/operational")
            / "multicycle_tracker_comparison.png"
        )
    else:
        output = Path(args.output)

    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if loaded_track_count == 0:
        plt.close(fig)
        raise RuntimeError(
            "No tracks loaded. Check storm names and input paths."
        )

    print(f"Loaded tracks: {loaded_track_count}")

    fig.savefig(
        output,
        dpi=180,
    )

    plt.close(fig)

    print(f"Created: {output}")


if __name__ == "__main__":
    main()
