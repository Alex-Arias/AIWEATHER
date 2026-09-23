"""
Plot multi-cycle operational WuDuan and Vitart tracks.

Rows
----
Odalys
Polo

Columns
-------
20 Sep 2026 12 UTC
21 Sep 2026 12 UTC
22 Sep 2026 12 UTC

Each panel contains the available GraphCast, AIFS2, Pangu3,
and Pangu6 tracker paths for one storm and initialization cycle.

The first point of each selected path is marked and annotated with
its forecast lead. Missing tracker detections are explicitly shown
as NO TRACK in the corresponding model list.

This script reads existing operational tracker CSV files only.
It does not rerun Earth2Studio and does not use IBTrACS.
"""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pandas as pd


CYCLES = [
    "20260920T120000",
    "20260921T120000",
    "20260922T120000",
]

STORMS = [
    "Odalys",
    "Polo",
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

DOMAINS = {
    "Odalys": (
        -140.0,
        -110.0,
        8.0,
        34.0,
    ),
    "Polo": (
        -125.0,
        -90.0,
        8.0,
        34.0,
    ),
}


def longitude_180(longitude):
    """
    Convert longitude from 0-360 degrees to -180-180 degrees.
    """
    return (
        (longitude + 180.0) % 360.0
    ) - 180.0


def tracker_path(
    *,
    storm,
    cycle,
    model,
    tracker,
):
    return (
        Path("results/operational")
        / f"{storm.lower()}_{cycle}"
        / model
        / f"{model}_{tracker}_track.csv"
    )


def load_track(
    *,
    storm,
    cycle,
    model,
    tracker,
):
    path = tracker_path(
        storm=storm,
        cycle=cycle,
        model=model,
        tracker=tracker,
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

    missing = required.difference(
        df.columns
    )

    if missing:
        raise ValueError(
            f"{path} is missing columns: "
            f"{sorted(missing)}"
        )

    df = df.copy()

    df["longitude_180"] = longitude_180(
        df["longitude"].astype(float)
    )

    df["latitude"] = df[
        "latitude"
    ].astype(float)

    df["lead_time_hours"] = df[
        "lead_time_hours"
    ].astype(float)

    return df


def make_figure(
    tracker,
    output,
):
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature

    projection = ccrs.PlateCarree()

    fig, axes = plt.subplots(
        2,
        3,
        figsize=(17, 10),
        subplot_kw={
            "projection": projection,
        },
    )

    for row, storm in enumerate(STORMS):

        (
            lon_min,
            lon_max,
            lat_min,
            lat_max,
        ) = DOMAINS[storm]

        for col, cycle in enumerate(CYCLES):

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

            # Avoid duplicated latitude labels between columns.
            if col > 0:
                gridlines.left_labels = False

            missing_models = []

            for model in MODELS:

                df = load_track(
                    storm=storm,
                    cycle=cycle,
                    model=model,
                    tracker=tracker,
                )

                if df is None:
                    missing_models.append(
                        model
                    )
                    continue

                color = MODEL_COLORS[
                    model
                ]

                ax.plot(
                    df["longitude_180"],
                    df["latitude"],
                    linewidth=2.0,
                    color=color,
                    transform=projection,
                    zorder=3,
                )

                first = df.iloc[0]

                ax.scatter(
                    first["longitude_180"],
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
                        first[
                            "lead_time_hours"
                        ]
                    )
                )

                ax.annotate(
                    f"+{first_lead} h",
                    xy=(
                        first[
                            "longitude_180"
                        ],
                        first["latitude"],
                    ),
                    xytext=(5, 5),
                    textcoords="offset points",
                    fontsize=7.5,
                    color=color,
                    transform=projection,
                    zorder=6,
                )

            timestamp = pd.to_datetime(
                cycle,
                format="%Y%m%dT%H%M%S",
            )

            if row == 0:
                ax.set_title(
                    timestamp.strftime(
                        "%d Sep 2026\n12 UTC"
                    ),
                    fontsize=12,
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
                missing_text = (
                    "NO TRACK\n"
                    + ", ".join(
                        missing_models
                    )
                )

                ax.text(
                    0.02,
                    0.02,
                    missing_text,
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

    model_handles = [
        Line2D(
            [0],
            [0],
            color=MODEL_COLORS[model],
            linewidth=2.2,
            marker="o",
            markersize=5,
            label=model,
        )
        for model in MODELS
    ]

    fig.legend(
        handles=model_handles,
        title="Forecast model",
        loc="lower center",
        ncol=4,
        frameon=True,
        bbox_to_anchor=(
            0.5,
            0.015,
        ),
    )

    tracker_title = {
        "wuduan": "WuDuan",
        "vitart": "Vitart",
    }.get(
        tracker,
        tracker,
    )

    fig.suptitle(
        "AIWeather Operational "
        f"{tracker_title} Track Evolution\n"
        "Selected forecast-only tracker paths",
        fontsize=16,
        y=0.975,
    )

    # Explicit layout rather than bbox_inches="tight":
    # Cartopy GeoAxes behaved more reliably this way in the
    # existing multicycle operational plotting workflow.
    fig.subplots_adjust(
        left=0.07,
        right=0.985,
        bottom=0.10,
        top=0.88,
        wspace=0.08,
        hspace=0.16,
    )

    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fig.savefig(
        output,
        dpi=180,
    )

    plt.close(fig)

    print(
        f"Created: {output}"
    )


def main():

    output_dir = Path(
        "results/operational"
    )

    for tracker in (
        "wuduan",
        "vitart",
    ):
        make_figure(
            tracker,
            output_dir
            / (
                f"multicycle_"
                f"{tracker}_tracks.png"
            ),
        )


if __name__ == "__main__":
    main()
