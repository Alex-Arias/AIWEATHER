#!/usr/bin/env python3

"""
Compare exported AIWeather operational tropical-cyclone tracks.

This script performs a model-to-model comparison of existing-TC tracks
previously exported by AIWeather. It does not require a best-track
reference and therefore reports model-to-model track *separation*, not
forecast track error.

Products
--------
model_track_metrics.csv
    Model-specific track and intensity statistics.

pairwise_track_separation.csv
    Pairwise model separation at exact common forecast lead times.

pairwise_track_summary.csv
    Summary statistics for each model pair.

track_map.png
    Geographic forecast-track comparison.

track_separation.png
    Pairwise track separation versus forecast lead time.

pressure_evolution.png
    Minimum sea-level pressure versus forecast lead time.

wind_evolution.png
    Maximum 10-m wind versus forecast lead time.
"""

from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path

import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from aiweather.plotting.tracks import normalize_longitude
from aiweather.tracking.comparison import compare_tracks
from aiweather.tracking.records import TrackRecord


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compare exported operational tropical-cyclone "
            "tracks from multiple AIWeather models."
        )
    )

    parser.add_argument(
        "--input-dir",
        required=True,
        type=Path,
        help=(
            "Directory containing *_track.csv and "
            "*_track.provenance.json files."
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Output directory. Defaults to "
            "<input-dir>/comparison."
        ),
    )

    parser.add_argument(
        "--title",
        default=None,
        help="Optional figure title prefix.",
    )

    return parser


def _optional_float(value) -> float | None:
    """Convert a scalar to float, preserving missing values as None."""
    if pd.isna(value):
        return None

    return float(value)


def _optional_string(value) -> str | None:
    """Convert a scalar to string, preserving missing values as None."""
    if pd.isna(value):
        return None

    text = str(value)

    if not text:
        return None

    return text


def load_provenance(path: Path) -> dict:
    """Read one operational-track provenance file."""
    with path.open(
        "r",
        encoding="utf-8",
    ) as handle:
        return json.load(handle)


def dataframe_to_records(
    dataframe: pd.DataFrame,
) -> list[TrackRecord]:
    """Convert an exported operational-track dataframe to TrackRecords."""
    records = []

    for row in dataframe.itertuples(
        index=False,
    ):
        records.append(
            TrackRecord(
                lead_time_hours=int(
                    row.lead_time_hours
                ),
                valid_time=np.datetime64(
                    row.valid_time
                ),
                latitude=float(
                    row.latitude
                ),
                longitude=float(
                    row.longitude
                ),
                pressure=float(
                    row.pressure
                ),
                pressure_units=(
                    _optional_string(
                        row.pressure_units
                    )
                ),
                max_wind=(
                    _optional_float(
                        row.max_wind
                    )
                ),
                wind_units=(
                    _optional_string(
                        row.wind_units
                    )
                ),
                distance_km=(
                    _optional_float(
                        row.distance_km
                    )
                ),
                translation_speed_kmh=(
                    _optional_float(
                        row.translation_speed_kmh
                    )
                ),
                bearing_degrees=(
                    _optional_float(
                        row.bearing_degrees
                    )
                ),
                cumulative_distance_km=float(
                    row.cumulative_distance_km
                ),
            )
        )

    return records


def discover_tracks(
    input_dir: Path,
) -> dict[str, dict]:
    """Discover operational track/provenance pairs."""
    provenance_paths = sorted(
        input_dir.rglob(
            "*_track.provenance.json"
        )
    )

    if not provenance_paths:
        raise FileNotFoundError(
            "No *_track.provenance.json files "
            f"found in {input_dir}"
        )

    tracks = {}

    for provenance_path in provenance_paths:
        track_path = (
            provenance_path.with_name(
                provenance_path.name.replace(
                    ".provenance.json",
                    ".csv",
                )
            )
        )

        if not track_path.exists():
            raise FileNotFoundError(
                "Missing operational track CSV: "
                f"{track_path}"
            )

        provenance = load_provenance(
            provenance_path
        )

        model = provenance[
            "forecast"
        ]["model_name"]

        dataframe = pd.read_csv(
            track_path
        )

        if dataframe.empty:
            print(
                f"Skipping empty track: "
                f"{track_path}"
            )
            continue

        dataframe[
            "valid_time"
        ] = pd.to_datetime(
            dataframe["valid_time"]
        )

        records = dataframe_to_records(
            dataframe
        )

        tracks[model] = {
            "records": records,
            "dataframe": dataframe,
            "provenance": provenance,
            "track_path": track_path,
            "provenance_path": provenance_path,
        }

    if not tracks:
        raise ValueError(
            "No non-empty operational tracks "
            "were found."
        )

    return tracks


def validate_common_case(
    tracks: dict[str, dict],
) -> None:
    """
    Verify that all tracks describe the same storm and initialization.
    """
    storm_ids = {
        item["provenance"]["storm"]["id"]
        for item in tracks.values()
    }

    initialization_times = {
        item["provenance"]["forecast"][
            "initialization_time"
        ]
        for item in tracks.values()
    }

    if len(storm_ids) != 1:
        raise ValueError(
            "Tracks contain different storm IDs: "
            f"{sorted(storm_ids)}"
        )

    if len(initialization_times) != 1:
        raise ValueError(
            "Tracks contain different initialization "
            f"times: {sorted(initialization_times)}"
        )


def build_model_metrics(
    tracks: dict[str, dict],
) -> pd.DataFrame:
    """Build model-specific track and intensity statistics."""
    rows = []

    for model, item in sorted(
        tracks.items()
    ):
        dataframe = item[
            "dataframe"
        ]

        provenance = item[
            "provenance"
        ]

        first = dataframe.iloc[0]
        last = dataframe.iloc[-1]

        pressure_index = dataframe[
            "pressure"
        ].idxmin()

        pressure_row = dataframe.loc[
            pressure_index
        ]

        wind_index = dataframe[
            "max_wind"
        ].idxmax()

        wind_row = dataframe.loc[
            wind_index
        ]

        rows.append(
            {
                "storm_id": provenance[
                    "storm"
                ]["id"],
                "storm_name": provenance[
                    "storm"
                ]["name"],
                "model": model,
                "forecast_id": provenance[
                    "forecast"
                ]["forecast_id"],
                "initialization_time": (
                    provenance[
                        "forecast"
                    ][
                        "initialization_time"
                    ]
                ),
                "experiment_type": (
                    provenance[
                        "tracking"
                    ][
                        "experiment_type"
                    ]
                ),
                "number_of_points": len(
                    dataframe
                ),
                "first_lead_time_hours": int(
                    first[
                        "lead_time_hours"
                    ]
                ),
                "last_lead_time_hours": int(
                    last[
                        "lead_time_hours"
                    ]
                ),
                "first_latitude": float(
                    first["latitude"]
                ),
                "first_longitude": float(
                    normalize_longitude(
                        first["longitude"]
                    )
                ),
                "last_latitude": float(
                    last["latitude"]
                ),
                "last_longitude": float(
                    normalize_longitude(
                        last["longitude"]
                    )
                ),
                "minimum_pressure_hpa": (
                    float(
                        pressure_row[
                            "pressure"
                        ]
                    )
                    / 100.0
                ),
                (
                    "minimum_pressure_"
                    "lead_time_hours"
                ): int(
                    pressure_row[
                        "lead_time_hours"
                    ]
                ),
                "minimum_pressure_time": (
                    pressure_row[
                        "valid_time"
                    ]
                ),
                "maximum_wind_mps": float(
                    wind_row[
                        "max_wind"
                    ]
                ),
                (
                    "maximum_wind_"
                    "lead_time_hours"
                ): int(
                    wind_row[
                        "lead_time_hours"
                    ]
                ),
                "maximum_wind_time": (
                    wind_row[
                        "valid_time"
                    ]
                ),
                "cumulative_distance_km": float(
                    last[
                        "cumulative_distance_km"
                    ]
                ),
            }
        )

    return pd.DataFrame(
        rows
    )


def build_pairwise_comparisons(
    tracks: dict[str, dict],
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
]:
    """
    Compare all model pairs at exact common forecast lead times.
    """
    detailed_tables = []
    summary_rows = []

    for model_a, model_b in combinations(
        sorted(tracks),
        2,
    ):
        comparison = compare_tracks(
            tracks[model_a]["records"],
            tracks[model_b]["records"],
            tracker_a=model_a,
            tracker_b=model_b,
        )

        table = comparison.table.copy()

        if table.empty:
            continue

        pair = (
            f"{model_a}_vs_{model_b}"
        )

        # compare_tracks() uses "track_error_km"
        # internally. Here neither model is truth, so
        # expose the scientifically correct term.
        table = table.rename(
            columns={
                "track_error_km":
                    "track_separation_km",
            }
        )

        table.insert(
            0,
            "pair",
            pair,
        )

        table.insert(
            1,
            "model_a",
            model_a,
        )

        table.insert(
            2,
            "model_b",
            model_b,
        )

        table[
            "pressure_difference_hpa"
        ] = (
            table[
                "pressure_difference"
            ]
            / 100.0
        )

        detailed_tables.append(
            table
        )

        summary_rows.append(
            {
                "pair": pair,
                "model_a": model_a,
                "model_b": model_b,
                (
                    "number_of_common_times"
                ): comparison.overlap_count,
                (
                    "first_common_lead_hours"
                ): int(
                    table[
                        "lead_time_hours"
                    ].min()
                ),
                (
                    "last_common_lead_hours"
                ): int(
                    table[
                        "lead_time_hours"
                    ].max()
                ),
                "mean_separation_km": (
                    comparison.
                    mean_track_error_km
                ),
                "rmse_separation_km": (
                    comparison.
                    rmse_track_error_km
                ),
                "median_separation_km": (
                    comparison.
                    median_track_error_km
                ),
                "maximum_separation_km": (
                    comparison.
                    maximum_track_error_km
                ),
            }
        )

    if detailed_tables:
        detailed = pd.concat(
            detailed_tables,
            ignore_index=True,
        )
    else:
        detailed = pd.DataFrame()

    summary = pd.DataFrame(
        summary_rows
    )

    return detailed, summary


def longitude_near_reference(
    longitude: float | np.ndarray,
    reference: float,
):
    """Place longitude on the 360-degree branch nearest a reference."""
    values = np.asarray(
        longitude,
        dtype=float,
    )

    adjusted = (
        reference
        + (
            values - reference + 180.0
        ) % 360.0
        - 180.0
    )

    if np.ndim(longitude) == 0:
        return float(adjusted)

    return adjusted


def plot_track_map(
    tracks: dict[str, dict],
    output_path: Path,
    title: str,
) -> None:
    """Plot all model forecast tracks and the initialization seed."""
    geographic_crs = (
        ccrs.PlateCarree()
    )

    figure = plt.figure(
        figsize=(10, 7)
    )

    geographic_crs = ccrs.PlateCarree()

    first_item = next(
        iter(tracks.values())
    )

    seed = first_item[
        "provenance"
    ]["seed"]

    seed_longitude = normalize_longitude(
        seed["longitude"]
    )

    seed_latitude = float(
        seed["latitude"]
    )

    map_crs = ccrs.PlateCarree(
        central_longitude=(
            180.0
            if abs(seed_longitude) > 150.0
            else 0.0
        )
    )

    figure = plt.figure(
        figsize=(10, 7)
    )

    ax = figure.add_subplot(
        1,
        1,
        1,
        projection=map_crs,
    )

    all_longitudes = []
    all_latitudes = []

    for model, item in sorted(
        tracks.items()
    ):
        records = item[
            "records"
        ]

        longitude = longitude_near_reference(
            np.asarray(
                [
                    normalize_longitude(
                        record.longitude
                    )
                    for record in records
                ],
                dtype=float,
            ),
            seed_longitude,
        )


        latitude = np.asarray(
            [
                record.latitude
                for record in records
            ],
            dtype=float,
        )

        all_longitudes.extend(
            longitude.tolist()
        )

        all_latitudes.extend(
            latitude.tolist()
        )

        ax.plot(
            longitude,
            latitude,
            marker="o",
            markersize=3,
            linewidth=1.5,
            label=model,
            transform=geographic_crs,
        )

        ax.scatter(
            longitude[0],
            latitude[0],
            marker="s",
            s=55,
            transform=geographic_crs,
        )

        ax.scatter(
            longitude[-1],
            latitude[-1],
            marker="X",
            s=65,
            transform=geographic_crs,
        )

    ax.scatter(
        seed_longitude,
        seed_latitude,
        marker="*",
        s=180,
        label="NHC initialization seed",
        transform=geographic_crs,
        zorder=10,
    )

    all_longitudes.append(
        seed_longitude
    )

    all_latitudes.append(
        seed_latitude
    )

    lon_min = min(all_longitudes) - 3.0
    lon_max = max(all_longitudes) + 3.0
    lat_min = min(all_latitudes) - 3.0
    lat_max = max(all_latitudes) + 3.0

    ax.set_extent(
        [
            lon_min,
            lon_max,
            lat_min,
            lat_max,
        ],
        crs=geographic_crs,
    )

    ax.coastlines(
        resolution="110m"
    )

    from cartopy.mpl.gridliner import (
        LATITUDE_FORMATTER,
        LONGITUDE_FORMATTER,
    )

    longitude_ticks = np.arange(
        np.floor(lon_min / 10.0) * 10.0,
        np.ceil(lon_max / 10.0) * 10.0 + 1.0,
        10.0,
    )

    latitude_ticks = np.arange(
        np.floor(lat_min / 5.0) * 5.0,
        np.ceil(lat_max / 5.0) * 5.0 + 1.0,
        5.0,
    )

    ax.set_xticks(
        longitude_ticks,
        crs=geographic_crs,
    )

    ax.set_yticks(
        latitude_ticks,
        crs=geographic_crs,
    )

    if abs(seed_longitude) > 150.0:
        longitude_labels = []

        for longitude in longitude_ticks:
            normalized = (
                longitude + 180.0
            ) % 360.0 - 180.0

            if np.isclose(
                abs(normalized),
                180.0,
            ):
                label = "180°"
            elif np.isclose(
                normalized,
                0.0,
            ):
                label = "0°"
            elif normalized > 0.0:
                label = (
                    f"{abs(normalized):g}°E"
                )
            else:
                label = (
                    f"{abs(normalized):g}°W"
                )

            longitude_labels.append(
                label
            )

        ax.set_xticklabels(
            longitude_labels
        )

    else:
        ax.xaxis.set_major_formatter(
            LONGITUDE_FORMATTER
        )

    ax.yaxis.set_major_formatter(
        LATITUDE_FORMATTER
    )

    ax.grid(
        alpha=0.3,
    )

    ax.set_title(
        title
    )

    ax.legend()

    figure.tight_layout()

    figure.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(
        figure
    )


def plot_separation(
    dataframe: pd.DataFrame,
    output_path: Path,
    title: str,
) -> None:
    """Plot pairwise model track separation."""
    if dataframe.empty:
        return

    figure, ax = plt.subplots(
        figsize=(10, 6)
    )

    for pair, group in dataframe.groupby(
        "pair",
        sort=True,
    ):
        ax.plot(
            group[
                "lead_time_hours"
            ],
            group[
                "track_separation_km"
            ],
            marker="o",
            markersize=3,
            linewidth=1.5,
            label=pair.replace(
                "_vs_",
                " vs ",
            ),
        )

    ax.set_xlabel(
        "Forecast lead time (h)"
    )

    ax.set_ylabel(
        "Model-to-model track separation (km)"
    )

    ax.set_title(
        title
    )

    ax.grid(
        alpha=0.3
    )

    ax.legend()

    figure.tight_layout()

    figure.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(
        figure
    )


def plot_pressure(
    tracks: dict[str, dict],
    output_path: Path,
    title: str,
) -> None:
    """Plot minimum pressure along each model track."""
    figure, ax = plt.subplots(
        figsize=(10, 6)
    )

    for model, item in sorted(
        tracks.items()
    ):
        dataframe = item[
            "dataframe"
        ]

        ax.plot(
            dataframe[
                "lead_time_hours"
            ],
            dataframe[
                "pressure"
            ]
            / 100.0,
            marker="o",
            markersize=3,
            linewidth=1.5,
            label=model,
        )

    ax.set_xlabel(
        "Forecast lead time (h)"
    )

    ax.set_ylabel(
        "Minimum sea-level pressure (hPa)"
    )

    ax.set_title(
        title
    )

    ax.grid(
        alpha=0.3
    )

    ax.legend()

    figure.tight_layout()

    figure.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(
        figure
    )


def plot_wind(
    tracks: dict[str, dict],
    output_path: Path,
    title: str,
) -> None:
    """Plot maximum local 10-m wind along each model track."""
    figure, ax = plt.subplots(
        figsize=(10, 6)
    )

    for model, item in sorted(
        tracks.items()
    ):
        dataframe = item[
            "dataframe"
        ]

        ax.plot(
            dataframe[
                "lead_time_hours"
            ],
            dataframe[
                "max_wind"
            ],
            marker="o",
            markersize=3,
            linewidth=1.5,
            label=model,
        )

    ax.set_xlabel(
        "Forecast lead time (h)"
    )

    ax.set_ylabel(
        "Maximum 10-m wind speed (m s$^{-1}$)"
    )

    ax.set_title(
        title
    )

    ax.grid(
        alpha=0.3
    )

    ax.legend()

    figure.tight_layout()

    figure.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(
        figure
    )


def main() -> None:
    args = build_parser().parse_args()

    input_dir = args.input_dir

    output_dir = (
        args.output_dir
        if args.output_dir is not None
        else input_dir / "comparison"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    tracks = discover_tracks(
        input_dir
    )

    validate_common_case(
        tracks
    )

    first_item = next(
        iter(tracks.values())
    )

    provenance = first_item[
        "provenance"
    ]

    storm_name = provenance[
        "storm"
    ]["name"]

    initialization_time = provenance[
        "forecast"
    ]["initialization_time"]

    experiment_type = provenance[
        "tracking"
    ]["experiment_type"]

    if args.title is None:
        base_title = (
            f"{storm_name} "
            f"{initialization_time}"
        )
    else:
        base_title = args.title

    model_metrics = (
        build_model_metrics(
            tracks
        )
    )

    (
        pairwise_separation,
        pairwise_summary,
    ) = build_pairwise_comparisons(
        tracks
    )

    model_metrics.to_csv(
        output_dir
        / "model_track_metrics.csv",
        index=False,
    )

    pairwise_separation.to_csv(
        output_dir
        / "pairwise_track_separation.csv",
        index=False,
    )

    pairwise_summary.to_csv(
        output_dir
        / "pairwise_track_summary.csv",
        index=False,
    )

    plot_track_map(
        tracks,
        output_dir
        / "track_map.png",
        (
            f"{base_title}\n"
            "Pseudo-operational model track comparison"
        ),
    )

    plot_separation(
        pairwise_separation,
        output_dir
        / "track_separation.png",
        (
            f"{base_title}\n"
            "Model-to-model track separation"
        ),
    )

    plot_pressure(
        tracks,
        output_dir
        / "pressure_evolution.png",
        (
            f"{base_title}\n"
            "Minimum sea-level pressure"
        ),
    )

    plot_wind(
        tracks,
        output_dir
        / "wind_evolution.png",
        (
            f"{base_title}\n"
            "Maximum 10-m wind speed"
        ),
    )

    print("=" * 78)
    print(
        "AIWEATHER OPERATIONAL TC MODEL COMPARISON"
    )
    print("=" * 78)

    print(
        f"Storm          : "
        f"{provenance['storm']['id']} "
        f"{storm_name}"
    )

    print(
        f"Initialization : "
        f"{initialization_time}"
    )

    print(
        f"Experiment     : "
        f"{experiment_type}"
    )

    print(
        "Models         : "
        + ", ".join(
            sorted(tracks)
        )
    )

    print(
        f"Output         : "
        f"{output_dir}"
    )

    print()

    print(
        "MODEL METRICS"
    )

    print(
        "-" * 78
    )

    print(
        model_metrics[
            [
                "model",
                "number_of_points",
                "last_lead_time_hours",
                "minimum_pressure_hpa",
                (
                    "minimum_pressure_"
                    "lead_time_hours"
                ),
                "maximum_wind_mps",
                (
                    "maximum_wind_"
                    "lead_time_hours"
                ),
            ]
        ].to_string(
            index=False
        )
    )

    if not pairwise_summary.empty:
        print()

        print(
            "PAIRWISE TRACK SEPARATION"
        )

        print(
            "-" * 78
        )

        print(
            pairwise_summary[
                [
                    "pair",
                    "number_of_common_times",
                    "last_common_lead_hours",
                    "mean_separation_km",
                    "rmse_separation_km",
                    "maximum_separation_km",
                ]
            ].to_string(
                index=False
            )
        )

    print()

    print(
        "Note: pairwise distances are "
        "model-to-model separation, not "
        "verification error."
    )


if __name__ == "__main__":
    main()