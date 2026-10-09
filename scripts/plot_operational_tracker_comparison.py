"""
Plot operational tropical-cyclone tracks by tracker.

Rows represent storms and columns represent tracking systems:
Native, WuDuan, and Vitart.

Each panel compares the available GraphCast, AIFS2, Pangu3, and
Pangu6 forecast tracks for a common forecast initialization.

The script reads existing operational track CSV files only. It does
not rerun a tracker and does not use IBTrACS.
"""

import argparse
from datetime import datetime, timedelta
from pathlib import Path

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

from aiweather.plotting.tracks import normalize_longitude
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

DATE_LABEL_OFFSETS = {
    "graphcast": (5, 7),
    "aifs2": (5, -12),
    "pangu3": (-32, 7),
    "pangu6": (-32, -12),
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
    "INVEST 92E": (-125.0, -90.0, 5.0, 30.0),
    "Simon": (-125.0, -90.0, 5.0, 30.0),
}


def storm_slug(name):
    """Return the canonical operational directory slug for a storm."""
    return (
        name.lower()
        .replace(" ", "_")
    )


def track_path(
    storm,
    model,
    tracker,
    init,
    experiment_slug=None,
):
    slug = (
        storm_slug(storm)
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

    df["longitude_plot"] = df[
        "longitude"
    ].astype(float).map(normalize_longitude)

    df["latitude"] = df[
        "latitude"
    ].astype(float)

    df["lead_time_hours"] = df[
        "lead_time_hours"
    ].astype(float)

    return df


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Plot operational tropical-cyclone tracks "
            "for multiple trackers and forecast models."
        )
    )

    parser.add_argument(
        "--init",
        required=True,
        help=(
            "Forecast initialization in YYYYMMDDTHHMMSS "
            "format."
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
            "Persistent experiment directory slug. "
            "Defaults to the normalized storm name."
        ),
    )

    parser.add_argument(
        "--show-utc-dates",
        action="store_true",
        help="Annotate 24-hour track markers with UTC dates.",
    )

    parser.add_argument(
        "--show-translation-speed",
        action="store_true",
        help="Add translation-speed diagnostics below the maps.",
    )

    parser.add_argument(
        "--focused-domain",
        action="store_true",
        help=(
            "Use a focused geographic domain for Simon. "
            "Other storms retain their default domains."
        ),
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

    try:
        init_time = datetime.strptime(
            args.init,
            "%Y%m%dT%H%M%S",
        )
    except ValueError as exc:
        raise ValueError(
            "--init must use YYYYMMDDTHHMMSS format"
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
        fig = plt.figure(
            figsize=(18, 11),
        )

        grid = fig.add_gridspec(
            len(storms) + 1,
            len(TRACKERS),
            height_ratios=[2.6] * len(storms) + [1.5],
            hspace=0.18,
        )

        axes = np.array([
            [
                fig.add_subplot(
                    grid[row, col],
                    projection=projection,
                )
                for col in range(len(TRACKERS))
            ]
            for row in range(len(storms))
        ], dtype=object)

        speed_axes = [
            fig.add_subplot(grid[-1, col])
            for col in range(len(TRACKERS))
        ]

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

    for row, storm in enumerate(storms):
        (
            lon_min,
            lon_max,
            lat_min,
            lat_max,
        ) = DOMAINS[storm]

        if args.focused_domain and storm == "Simon":
            lon_min = -110.0
            lon_max = -99.0
            lat_min = 13.0
            lat_max = 26.0

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

            gridlines.top_labels = False
            gridlines.right_labels = False

            if col > 0:
                gridlines.left_labels = False

            missing_models = []

            for model in MODELS:
                df = load_track(
                    storm,
                    model,
                    tracker,
                    args.init,
                    experiment_slug=args.experiment_slug,
                )

                if df is None:
                    missing_models.append(
                        MODEL_LABELS[model]
                    )
                    continue

                color = MODEL_COLORS[model]

                if args.show_translation_speed:
                    speed_df = calculate_translation_speed(df)

                    speed_axes[col].plot(
                        speed_df["lead_time_hours"],
                        speed_df["translation_speed_kmh"],
                        color=color,
                        linewidth=1.8,
                        marker=".",
                        markersize=4,
                        label=MODEL_LABELS[model],
                    )

                    if tracker == "native":
                        high_speed = speed_df[
                            speed_df["translation_speed_kmh"] > 30.0
                        ]

                        if not high_speed.empty:
                            speed_axes[col].scatter(
                                high_speed["lead_time_hours"],
                                high_speed["translation_speed_kmh"],
                                facecolors="none",
                                edgecolors=color,
                                s=65,
                                linewidths=1.5,
                                zorder=8,
                            )

                ax.plot(
                    df["longitude_plot"],
                    df["latitude"],
                    linewidth=2.0,
                    color=color,
                    transform=projection,
                    zorder=3,
                )

                # -------------------------------------------------
                # Forecast-time markers
                # -------------------------------------------------

                first = df.iloc[0]

                # Mark the first available tracker point.
                ax.scatter(
                    first["longitude_plot"],
                    first["latitude"],
                    marker="o",
                    s=42,
                    color=color,
                    edgecolor="white",
                    linewidth=0.6,
                    transform=projection,
                    zorder=5,
                )

                first_lead = int(
                    round(
                        first["lead_time_hours"]
                    )
                )

                # Annotate only delayed tracker starts.
                # A track beginning at forecast initialization (+0 h)
                # does not need a redundant text label.
                if first_lead > 0:
                    ax.annotate(
                        f"+{first_lead} h",
                        xy=(
                            first["longitude_plot"],
                            first["latitude"],
                        ),
                        xytext=(5, 5),
                        textcoords="offset points",
                        fontsize=7.5,
                        color=color,
                        transform=projection,
                        zorder=6,
                    )

                # Add common 24-hour forecast markers:
                # +24, +48, +72, ... h.
                #
                # These are referenced to forecast initialization,
                # not to the first tracker detection, so marker
                # positions are directly comparable among models.
                lead = df["lead_time_hours"].round().astype(int)

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
                        s=20,
                        color=color,
                        edgecolor="white",
                        linewidth=0.45,
                        transform=projection,
                        zorder=4,
                    )

                    if args.show_utc_dates:
                        for _, point in marker_df.iterrows():
                            lead_hour = float(
                                point["lead_time_hours"]
                            )

                            valid_time = (
                                init_time
                                + timedelta(hours=lead_hour)
                            )

                            date_label = valid_time.strftime(
                                "%d %b"
                            )

                            ax.annotate(
                                date_label,
                                xy=(
                                    point["longitude_plot"],
                                    point["latitude"],
                                ),
                                xytext=DATE_LABEL_OFFSETS[model],
                                textcoords="offset points",
                                fontsize=6.5,
                                color=color,
                                transform=projection,
                                zorder=7,
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

            if missing_models:
                ax.text(
                    0.02,
                    0.02,
                    "NO TRACK\n"
                    + ", ".join(missing_models),
                    transform=ax.transAxes,
                    fontsize=8,
                    va="bottom",
                    ha="left",
                    bbox={
                        "facecolor": "white",
                        "alpha": 0.80,
                        "edgecolor": "0.65",
                    },
                    zorder=10,
                )

    if speed_axes is not None:
        for col, tracker in enumerate(TRACKERS):
            speed_ax = speed_axes[col]

            speed_ax.set_title(
                f"{TRACKER_LABELS[tracker]} translation speed",
                fontsize=11,
            )

            speed_ax.set_xlabel(
                "Forecast lead time (h)"
            )

            speed_ax.set_ylabel(
                "Translation speed (km/h)"
            )

            speed_ax.grid(
                True,
                alpha=0.3,
            )

            speed_ax.set_xlim(left=0)
            speed_ax.set_ylim(bottom=0)

    handles = [
        Line2D(
            [0],
            [0],
            color=MODEL_COLORS[model],
            linewidth=2.2,
            marker="o",
            markersize=5,
            label=MODEL_LABELS[model],
        )
        for model in MODELS
    ]

    fig.legend(
        handles=handles,
        title="Forecast model",
        loc="lower center",
        ncol=4,
        frameon=True,
        bbox_to_anchor=(0.5, 0.030),
    )

    fig.text(
        0.5,
        0.008,
        (
            "Map markers every 24 h; larger circle = first detection. "
            "Open speed circles: Native >30 km/h (diagnostic only)."
        ),
        ha="center",
        va="bottom",
        fontsize=9,
    )

    title_time = init_time.strftime(
        "%d %b %Y %H UTC"
    )

    fig.suptitle(
        "AIWeather Operational Track Comparison\n"
        f"{title_time} — forecast-only tracks",
        fontsize=16,
        y=0.975,
    )

    if args.show_translation_speed:
        fig.subplots_adjust(
            left=0.07,
            right=0.985,
            bottom=0.12,
            top=0.88,
            wspace=0.12,
            hspace=0.30,
        )
    else:
        fig.subplots_adjust(
            left=0.07,
            right=0.985,
            bottom=0.10,
            top=0.88,
            wspace=0.08,
            hspace=0.16,
        )

    if args.output is None:
        output = (
            Path("results/operational")
            / f"tracker_comparison_{args.init}.png"
        )
    else:
        output = Path(args.output)

    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fig.savefig(
        output,
        dpi=180,
    )

    plt.close(fig)

    print(f"Created: {output}")


if __name__ == "__main__":
    main()
