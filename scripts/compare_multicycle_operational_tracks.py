#!/usr/bin/env python3
"""
Compare native tropical-cyclone tracks across consecutive AIWeather
operational forecast cycles.

The comparison uses exported native-track CSV files only. Forecast
inference and TC detection are not rerun.

Track revisions are evaluated at exact common valid times, not at
common forecast lead times.

Outputs
-------
1. Per-cycle detection/genesis summary.
2. Consecutive-cycle track-revision statistics.
3. Common-valid-time pointwise revision table.
4. Multi-cycle track figure.
5. Detection-lead evolution figure.

The script accepts an arbitrary number of forecast cycles so the same
workflow can be rerun when new operational cycles become available.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


MODELS = (
    "graphcast",
    "aifs2",
    "pangu3",
    "pangu6",
)

STORMS = (
    "Odalys",
    "Polo",
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Compare native TC tracks across operational "
            "forecast cycles."
        )
    )

    parser.add_argument(
        "--cycles",
        nargs="+",
        required=True,
        help=(
            "Initialization cycles as YYYYMMDDTHHMMSS. "
            "Supply them in chronological order."
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "outputs/verification/"
            "multicycle_operational"
        ),
    )

    return parser.parse_args()


def lon180(values):
    values = np.asarray(
        values,
        dtype=float,
    )

    return (
        (values + 180.0) % 360.0
    ) - 180.0


def great_circle_distance_km(
    lat1,
    lon1,
    lat2,
    lon2,
):
    """
    Vectorized great-circle distance.
    """

    radius_km = 6371.0

    lat1 = np.radians(
        np.asarray(lat1, dtype=float)
    )
    lat2 = np.radians(
        np.asarray(lat2, dtype=float)
    )

    lon1 = np.radians(
        lon180(lon1)
    )
    lon2 = np.radians(
        lon180(lon2)
    )

    dlat = lat2 - lat1

    dlon = (
        lon2 - lon1 + np.pi
    ) % (2.0 * np.pi) - np.pi

    a = (
        np.sin(dlat / 2.0) ** 2
        + np.cos(lat1)
        * np.cos(lat2)
        * np.sin(dlon / 2.0) ** 2
    )

    return (
        2.0
        * radius_km
        * np.arcsin(
            np.sqrt(
                np.clip(a, 0.0, 1.0)
            )
        )
    )


def track_path(
    storm,
    cycle,
    model,
):
    return (
        Path("results/operational")
        / f"{storm.lower()}_{cycle}"
        / model
        / f"{model}_track.csv"
    )


def load_track(
    storm,
    cycle,
    model,
):
    path = track_path(
        storm,
        cycle,
        model,
    )

    if not path.exists():
        return None

    df = pd.read_csv(path)

    df["valid_time"] = pd.to_datetime(
        df["valid_time"]
    )

    df["longitude_180"] = lon180(
        df["longitude"]
    )

    return df.sort_values(
        "valid_time"
    ).reset_index(drop=True)


def build_detection_summary(
    tracks,
    cycles,
):
    rows = []

    for cycle in cycles:
        for storm in STORMS:
            for model in MODELS:

                df = tracks[
                    (storm, cycle, model)
                ]

                if df is None or df.empty:
                    rows.append(
                        {
                            "cycle": cycle,
                            "storm": storm,
                            "model": model,
                            "detected": False,
                            "genesis_lead_hours": np.nan,
                            "genesis_latitude": np.nan,
                            "genesis_longitude": np.nan,
                            "track_points": 0,
                            "last_lead_hours": np.nan,
                        }
                    )
                    continue

                first = df.iloc[0]
                last = df.iloc[-1]

                rows.append(
                    {
                        "cycle": cycle,
                        "storm": storm,
                        "model": model,
                        "detected": True,
                        "genesis_lead_hours": (
                            first["lead_time_hours"]
                        ),
                        "genesis_latitude": (
                            first["latitude"]
                        ),
                        "genesis_longitude": (
                            first["longitude_180"]
                        ),
                        "track_points": len(df),
                        "last_lead_hours": (
                            last["lead_time_hours"]
                        ),
                    }
                )

    return pd.DataFrame(rows)


def compare_pair(
    storm,
    model,
    earlier_cycle,
    later_cycle,
    earlier,
    later,
):
    if (
        earlier is None
        or later is None
        or earlier.empty
        or later.empty
    ):
        return None, None

    columns = [
        "valid_time",
        "lead_time_hours",
        "latitude",
        "longitude_180",
        "pressure",
        "max_wind",
    ]

    merged = earlier[
        columns
    ].merge(
        later[columns],
        on="valid_time",
        how="inner",
        suffixes=("_earlier", "_later"),
    )

    if merged.empty:
        return None, None

    merged["track_revision_km"] = (
        great_circle_distance_km(
            merged["latitude_earlier"],
            merged["longitude_180_earlier"],
            merged["latitude_later"],
            merged["longitude_180_later"],
        )
    )

    merged["pressure_revision_pa"] = (
        merged["pressure_later"]
        - merged["pressure_earlier"]
    )

    merged["wind_revision_ms"] = (
        merged["max_wind_later"]
        - merged["max_wind_earlier"]
    )

    merged.insert(
        0,
        "storm",
        storm,
    )

    merged.insert(
        1,
        "model",
        model,
    )

    merged.insert(
        2,
        "earlier_cycle",
        earlier_cycle,
    )

    merged.insert(
        3,
        "later_cycle",
        later_cycle,
    )

    distances = merged[
        "track_revision_km"
    ].to_numpy()

    summary = {
        "storm": storm,
        "model": model,
        "earlier_cycle": earlier_cycle,
        "later_cycle": later_cycle,
        "common_times": len(merged),
        "mean_track_revision_km": (
            np.nanmean(distances)
        ),
        "rmse_track_revision_km": (
            np.sqrt(
                np.nanmean(
                    distances ** 2
                )
            )
        ),
        "median_track_revision_km": (
            np.nanmedian(distances)
        ),
        "maximum_track_revision_km": (
            np.nanmax(distances)
        ),
        "mean_pressure_revision_pa": (
            np.nanmean(
                merged["pressure_revision_pa"]
            )
        ),
        "mean_absolute_pressure_revision_pa": (
            np.nanmean(
                np.abs(
                    merged["pressure_revision_pa"]
                )
            )
        ),
        "mean_wind_revision_ms": (
            np.nanmean(
                merged["wind_revision_ms"]
            )
        ),
        "mean_absolute_wind_revision_ms": (
            np.nanmean(
                np.abs(
                    merged["wind_revision_ms"]
                )
            )
        ),
    }

    return summary, merged


def make_track_figure(
    tracks,
    cycles,
    output,
):
    """
    Plot operational native tracks on geographic maps.

    Model is represented by color and forecast cycle by
    line style. The first native detection in each track
    is marked with a filled circle.
    """

    import cartopy.crs as ccrs
    import cartopy.feature as cfeature

    projection = ccrs.PlateCarree()

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(15, 6.5),
        subplot_kw={
            "projection": projection,
        },
    )

    # Fixed model colors across both storms.
    model_colors = {
        "graphcast": "C0",
        "aifs2": "C1",
        "pangu3": "C2",
        "pangu6": "C3",
    }

    # Different line style for each operational cycle.
    line_styles = [
        ":",
        "--",
        "-",
        "-.",
    ]

    # Domains chosen to retain the complete forecast
    # trajectories while showing the Mexico coastline.
    domains = {
        "Odalys": (
            -138.0,
            -112.0,
            10.0,
            34.0,
        ),
        "Polo": (
            -125.0,
            -95.0,
            10.0,
            34.0,
        ),
    }

    for ax, storm in zip(
        axes,
        STORMS,
    ):
        lon_min, lon_max, lat_min, lat_max = (
            domains[storm]
        )

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

        for model in MODELS:
            for cycle_index, cycle in enumerate(
                cycles
            ):
                df = tracks[
                    (storm, cycle, model)
                ]

                if df is None or df.empty:
                    continue

                linestyle = line_styles[
                    cycle_index
                    % len(line_styles)
                ]

                ax.plot(
                    df["longitude_180"],
                    df["latitude"],
                    linewidth=1.8,
                    linestyle=linestyle,
                    color=model_colors[model],
                    transform=projection,
                    zorder=3,
                )

                # First native detection.
                ax.scatter(
                    df.iloc[0]["longitude_180"],
                    df.iloc[0]["latitude"],
                    marker="o",
                    s=35,
                    color=model_colors[model],
                    edgecolor="white",
                    linewidth=0.5,
                    transform=projection,
                    zorder=5,
                )

        ax.set_title(
            storm,
            fontsize=14,
        )

    # Separate legends prevent color and cycle meaning
    # from becoming ambiguous.
    from matplotlib.lines import Line2D

    model_handles = [
        Line2D(
            [0],
            [0],
            color=model_colors[model],
            linewidth=2.0,
            label=model,
        )
        for model in MODELS
    ]

    cycle_handles = []

    for index, cycle in enumerate(cycles):
        timestamp = pd.to_datetime(
            cycle,
            format="%Y%m%dT%H%M%S",
        )

        cycle_handles.append(
            Line2D(
                [0],
                [0],
                color="black",
                linewidth=1.8,
                linestyle=line_styles[
                    index % len(line_styles)
                ],
                label=timestamp.strftime(
                    "%d Sep %H UTC"
                ),
            )
        )

    model_legend = axes[0].legend(
        handles=model_handles,
        title="Model",
        loc="upper left",
        frameon=True,
    )

    axes[0].add_artist(
        model_legend
    )

    axes[0].legend(
        handles=cycle_handles,
        title="Initialization",
        loc="lower left",
        frameon=True,
    )

    fig.suptitle(
        "AIWeather Operational Native Track Evolution",
        fontsize=16,
        y=0.97,
    )

    # Explicit layout is more reliable with Cartopy GeoAxes
    # than bbox_inches="tight".
    fig.subplots_adjust(
        left=0.06,
        right=0.98,
        bottom=0.08,
        top=0.88,
        wspace=0.12,
    )

    fig.savefig(
        output,
        dpi=180,
    )

    plt.close(fig)

def make_detection_figure(
    detection,
    cycles,
    output,
):
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(14, 5),
        sharey=True,
    )

    x = np.arange(
        len(cycles)
    )

    for ax, storm in zip(
        axes,
        STORMS,
    ):
        subset = detection[
            detection["storm"] == storm
        ]

        for model in MODELS:
            model_data = (
                subset[
                    subset["model"] == model
                ]
                .set_index("cycle")
                .reindex(cycles)
            )

            ax.plot(
                x,
                model_data[
                    "genesis_lead_hours"
                ],
                marker="o",
                linewidth=2.0,
                label=model,
            )

        ax.set_title(storm)
        ax.set_xticks(x)
        ax.set_xticklabels(
            [
                pd.to_datetime(
                    cycle,
                    format="%Y%m%dT%H%M%S",
                ).strftime("%d Sep")
                for cycle in cycles
            ]
        )

        ax.set_xlabel(
            "Forecast initialization"
        )

        ax.grid(
            True,
            alpha=0.3,
        )

    axes[0].set_ylabel(
        "Native detection lead time (h)"
    )

    axes[1].legend(
        title="Model",
    )

    fig.suptitle(
        "Operational Native TC Detection Evolution",
        fontsize=14,
    )

    fig.savefig(
        output,
        dpi=180,
        bbox_inches="tight",
    )

    plt.close(fig)


def main():
    args = parse_args()

    cycles = args.cycles

    if len(cycles) < 2:
        raise ValueError(
            "At least two cycles are required."
        )

    output_dir = args.output_dir

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    tracks = {}

    for storm in STORMS:
        for cycle in cycles:
            for model in MODELS:
                tracks[
                    (storm, cycle, model)
                ] = load_track(
                    storm,
                    cycle,
                    model,
                )

    detection = build_detection_summary(
        tracks,
        cycles,
    )

    detection_path = (
        output_dir
        / "native_detection_multicycle.csv"
    )

    detection.to_csv(
        detection_path,
        index=False,
    )

    summaries = []
    pointwise = []

    for earlier_cycle, later_cycle in zip(
        cycles[:-1],
        cycles[1:],
    ):
        for storm in STORMS:
            for model in MODELS:

                summary, merged = compare_pair(
                    storm,
                    model,
                    earlier_cycle,
                    later_cycle,
                    tracks[
                        (
                            storm,
                            earlier_cycle,
                            model,
                        )
                    ],
                    tracks[
                        (
                            storm,
                            later_cycle,
                            model,
                        )
                    ],
                )

                if summary is not None:
                    summaries.append(summary)

                if merged is not None:
                    pointwise.append(merged)

    summary_df = pd.DataFrame(
        summaries
    )

    summary_path = (
        output_dir
        / "consecutive_cycle_summary.csv"
    )

    summary_df.to_csv(
        summary_path,
        index=False,
    )

    if pointwise:
        pointwise_df = pd.concat(
            pointwise,
            ignore_index=True,
        )
    else:
        pointwise_df = pd.DataFrame()

    pointwise_path = (
        output_dir
        / "consecutive_cycle_pointwise.csv"
    )

    pointwise_df.to_csv(
        pointwise_path,
        index=False,
    )

    track_figure = (
        output_dir
        / "multicycle_native_tracks.png"
    )

    detection_figure = (
        output_dir
        / "native_detection_evolution.png"
    )

    make_track_figure(
        tracks,
        cycles,
        track_figure,
    )

    make_detection_figure(
        detection,
        cycles,
        detection_figure,
    )

    print("=" * 76)
    print("MULTICYCLE OPERATIONAL TRACK COMPARISON")
    print("=" * 76)

    print()
    print("Cycles:")
    for cycle in cycles:
        print(" ", cycle)

    print()
    print("Detection summary:")
    print(
        detection[
            [
                "cycle",
                "storm",
                "model",
                "detected",
                "genesis_lead_hours",
                "last_lead_hours",
            ]
        ].to_string(index=False)
    )

    print()
    print("=" * 76)
    print("CONSECUTIVE-CYCLE TRACK REVISIONS")
    print("=" * 76)

    if not summary_df.empty:
        display_columns = [
            "storm",
            "model",
            "earlier_cycle",
            "later_cycle",
            "common_times",
            "mean_track_revision_km",
            "rmse_track_revision_km",
            "maximum_track_revision_km",
            "mean_absolute_pressure_revision_pa",
            "mean_absolute_wind_revision_ms",
        ]

        print(
            summary_df[
                display_columns
            ].to_string(
                index=False,
                float_format=lambda value: (
                    f"{value:.2f}"
                ),
            )
        )

    print()
    print("Outputs:")
    print(" ", detection_path)
    print(" ", summary_path)
    print(" ", pointwise_path)
    print(" ", track_figure)
    print(" ", detection_figure)


if __name__ == "__main__":
    main()
