#!/usr/bin/env python3

"""
Compare two AIWeather operational tropical-cyclone forecast cycles.

Tracks from different initialization cycles are compared at exact common
valid times. The resulting distances are cycle-to-cycle forecast
displacements, not verification errors.

System spread is calculated across independent forecast systems at exact
common valid times. By default, Pangu6 is excluded from the spread
calculation when Pangu3 is present because Pangu3 and Pangu6 represent
different temporal-cadence configurations of the same Pangu system.
"""

from __future__ import annotations

import argparse
import importlib.util
from itertools import combinations
from pathlib import Path

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


TRACK_SCRIPT = (
    Path(__file__).resolve().parent
    / "compare_operational_tc_tracks.py"
)

SPEC = importlib.util.spec_from_file_location(
    "compare_operational_tc_tracks",
    TRACK_SCRIPT,
)

TRACK_MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(TRACK_MODULE)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compare two operational tropical-cyclone "
            "forecast cycles at exact common valid times."
        )
    )
    parser.add_argument(
        "--baseline-dir",
        required=True,
        type=Path,
    )
    parser.add_argument(
        "--later-dir",
        required=True,
        type=Path,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--spread-models",
        nargs="+",
        default=None,
        help=(
            "Models used for inter-system spread. "
            "Default: all common models."
        ),
    )
    parser.add_argument(
        "--title",
        default=None,
    )
    return parser


def great_circle_distance_km(
    lat1,
    lon1,
    lat2,
    lon2,
):
    """Great-circle distance in kilometres."""
    radius_km = 6371.0

    phi1 = np.radians(lat1)
    phi2 = np.radians(lat2)

    delta_phi = np.radians(
        np.asarray(lat2) - np.asarray(lat1)
    )

    delta_lon = np.radians(
        (
            np.asarray(lon2)
            - np.asarray(lon1)
            + 180.0
        )
        % 360.0
        - 180.0
    )

    value = (
        np.sin(delta_phi / 2.0) ** 2
        + np.cos(phi1)
        * np.cos(phi2)
        * np.sin(delta_lon / 2.0) ** 2
    )

    return (
        2.0
        * radius_km
        * np.arcsin(
            np.sqrt(
                np.clip(value, 0.0, 1.0)
            )
        )
    )


def validate_cycles(
    baseline: dict[str, dict],
    later: dict[str, dict],
) -> None:
    """Validate storm identity and initialization ordering."""
    TRACK_MODULE.validate_common_case(baseline)
    TRACK_MODULE.validate_common_case(later)

    baseline_ids = {
        item["provenance"]["storm"]["id"]
        for item in baseline.values()
    }

    later_ids = {
        item["provenance"]["storm"]["id"]
        for item in later.values()
    }

    if baseline_ids != later_ids:
        raise ValueError(
            "Baseline and later cycles describe "
            "different storm IDs."
        )

    baseline_init = pd.Timestamp(
        next(iter(baseline.values()))[
            "provenance"
        ]["forecast"]["initialization_time"]
    )

    later_init = pd.Timestamp(
        next(iter(later.values()))[
            "provenance"
        ]["forecast"]["initialization_time"]
    )

    if later_init <= baseline_init:
        raise ValueError(
            "Later initialization must occur after "
            "baseline initialization."
        )


def build_cycle_displacement(
    baseline: dict[str, dict],
    later: dict[str, dict],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compare each model between cycles at exact common valid times."""
    detailed_tables = []
    summary_rows = []

    common_models = sorted(
        set(baseline) & set(later)
    )

    for model in common_models:
        first = baseline[model]["dataframe"].copy()
        second = later[model]["dataframe"].copy()

        first["valid_time"] = pd.to_datetime(
            first["valid_time"]
        )
        second["valid_time"] = pd.to_datetime(
            second["valid_time"]
        )

        table = first.merge(
            second,
            on="valid_time",
            suffixes=("_baseline", "_later"),
        )

        if table.empty:
            continue

        table.insert(0, "model", model)

        table["delta_lead_hours"] = (
            table["lead_time_hours_baseline"]
            - table["lead_time_hours_later"]
        )

        table["delta_latitude_degrees"] = (
            table["latitude_later"]
            - table["latitude_baseline"]
        )

        table["delta_longitude_degrees"] = (
            (
                table["longitude_later"]
                - table["longitude_baseline"]
                + 180.0
            )
            % 360.0
            - 180.0
        )

        table["cycle_displacement_km"] = (
            great_circle_distance_km(
                table["latitude_baseline"],
                table["longitude_baseline"],
                table["latitude_later"],
                table["longitude_later"],
            )
        )

        detailed_tables.append(table)

        summary_rows.append(
            {
                "model": model,
                "number_of_common_times": len(table),
                "first_common_valid_time":
                    table["valid_time"].min(),
                "last_common_valid_time":
                    table["valid_time"].max(),
                "mean_cycle_displacement_km":
                    table["cycle_displacement_km"].mean(),
                "median_cycle_displacement_km":
                    table["cycle_displacement_km"].median(),
                "maximum_cycle_displacement_km":
                    table["cycle_displacement_km"].max(),
                "mean_delta_latitude_degrees":
                    table["delta_latitude_degrees"].mean(),
                "final_delta_latitude_degrees":
                    table["delta_latitude_degrees"].iloc[-1],
            }
        )

    if detailed_tables:
        detailed = pd.concat(
            detailed_tables,
            ignore_index=True,
        )
    else:
        detailed = pd.DataFrame()

    return detailed, pd.DataFrame(summary_rows)


def resolve_spread_models(
    baseline: dict[str, dict],
    later: dict[str, dict],
    requested: list[str] | None = None,
) -> list[str]:
    """Resolve systems used for inter-system spread."""
    common = sorted(
        set(baseline) & set(later)
    )

    if requested is None:
        return common

    models = list(dict.fromkeys(requested))

    missing = [
        model
        for model in models
        if model not in common
    ]

    if missing:
        raise ValueError(
            "Spread models must exist in both cycles. "
            f"Missing: {', '.join(missing)}"
        )

    if len(models) < 2:
        raise ValueError(
            "At least two spread models are required."
        )

    return models


def build_system_spread(
    tracks: dict[str, dict],
    cycle_label: str,
    models: list[str],
) -> pd.DataFrame:
    """Calculate mean pairwise system spread at exact common valid times."""
    if len(models) < 2:
        raise ValueError(
            "At least two spread models are required."
        )

    frames = {}

    for model in models:
        dataframe = tracks[model]["dataframe"].copy()

        dataframe["valid_time"] = pd.to_datetime(
            dataframe["valid_time"]
        )

        frames[model] = dataframe

    common_times = set(
        frames[models[0]]["valid_time"]
    )

    for model in models[1:]:
        common_times &= set(
            frames[model]["valid_time"]
        )

    rows = []

    for valid_time in sorted(common_times):
        points = {}

        for model in models:
            row = frames[model].loc[
                frames[model]["valid_time"]
                == valid_time
            ].iloc[0]

            points[model] = (
                float(row["latitude"]),
                float(row["longitude"]),
            )

        distances = []

        for model_a, model_b in combinations(
            models,
            2,
        ):
            distances.append(
                float(
                    great_circle_distance_km(
                        *points[model_a],
                        *points[model_b],
                    )
                )
            )

        rows.append(
            {
                "cycle": cycle_label,
                "valid_time": valid_time,
                "number_of_systems": len(models),
                "systems": ",".join(models),
                "mean_pairwise_spread_km":
                    float(np.mean(distances)),
                "median_pairwise_spread_km":
                    float(np.median(distances)),
                "maximum_pairwise_spread_km":
                    float(np.max(distances)),
            }
        )

    return pd.DataFrame(rows)


def summarize_system_spread(
    spread: pd.DataFrame,
) -> pd.DataFrame:
    """Summarize system spread by cycle."""
    rows = []

    for cycle, table in spread.groupby(
        "cycle",
        sort=False,
    ):
        rows.append(
            {
                "cycle": cycle,
                "number_of_common_times": len(table),
                "systems": table["systems"].iloc[0],
                "mean_pairwise_spread_km":
                    table["mean_pairwise_spread_km"].mean(),
                "median_pairwise_spread_km":
                    table["mean_pairwise_spread_km"].median(),
                "maximum_pairwise_spread_km":
                    table["maximum_pairwise_spread_km"].max(),
            }
        )

    return pd.DataFrame(rows)



def match_spread_valid_times(
    baseline_spread: pd.DataFrame,
    later_spread: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Restrict two spread series to their exact common valid times."""
    common_times = sorted(
        set(baseline_spread["valid_time"])
        & set(later_spread["valid_time"])
    )

    baseline_matched = baseline_spread[
        baseline_spread["valid_time"].isin(
            common_times
        )
    ].copy()

    later_matched = later_spread[
        later_spread["valid_time"].isin(
            common_times
        )
    ].copy()

    baseline_matched["cycle"] = (
        "baseline_matched"
    )
    later_matched["cycle"] = (
        "later_matched"
    )

    return (
        baseline_matched.reset_index(
            drop=True
        ),
        later_matched.reset_index(
            drop=True
        ),
    )



def model_display_name(model: str) -> str:
    """Return presentation label for a forecast system."""
    labels = {
        "aifs2": "AIFS2",
        "graphcast": "GraphCast",
        "pangu3": "Pangu3 (3-h cadence)",
        "pangu6": "Pangu6 (6-h cadence)",
    }
    return labels.get(model, model)


def plot_cycle_displacement(
    displacement: pd.DataFrame,
    output_path: Path,
    title: str,
) -> None:
    """Plot cycle-to-cycle displacement at exact common valid times."""
    figure, ax = plt.subplots(
        figsize=(10, 6)
    )

    for model, table in displacement.groupby(
        "model",
        sort=True,
    ):
        table = table.sort_values(
            "valid_time"
        )

        ax.plot(
            table["valid_time"],
            table["cycle_displacement_km"],
            marker="o",
            markersize=3,
            linewidth=1.5,
            label=model_display_name(model),
        )

    ax.set_ylabel(
        "Cycle displacement (km)"
    )
    ax.set_xlabel(
        "Valid time (UTC)"
    )
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend()

    ax.xaxis.set_major_formatter(
        mdates.DateFormatter("%d %b")
    )

    figure.autofmt_xdate()
    figure.tight_layout()

    figure.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(figure)


def plot_matched_system_spread(
    matched_spread: pd.DataFrame,
    output_path: Path,
    title: str,
) -> None:
    """Plot matched-window independent-system spread."""
    figure, ax = plt.subplots(
        figsize=(10, 6)
    )

    labels = {
        "baseline_matched": "Baseline cycle",
        "later_matched": "Later cycle",
    }

    for cycle, table in matched_spread.groupby(
        "cycle",
        sort=False,
    ):
        table = table.sort_values(
            "valid_time"
        )

        label = labels.get(
            cycle,
            cycle,
        )

        line = ax.plot(
            table["valid_time"],
            table["mean_pairwise_spread_km"],
            marker="o",
            markersize=3,
            linewidth=1.5,
            label=label,
        )[0]

        mean_value = table[
            "mean_pairwise_spread_km"
        ].mean()

        ax.axhline(
            mean_value,
            linestyle="--",
            linewidth=1.2,
            color=line.get_color(),
            label=(
                f"{label} mean "
                f"({mean_value:.0f} km)"
            ),
        )

    cycle_means = (
        matched_spread.groupby("cycle")[
            "mean_pairwise_spread_km"
        ]
        .mean()
    )

    if (
        "baseline_matched" in cycle_means
        and "later_matched" in cycle_means
        and cycle_means["baseline_matched"] > 0.0
    ):
        reduction = 100.0 * (
            1.0
            - cycle_means["later_matched"]
            / cycle_means["baseline_matched"]
        )

        ax.text(
            0.98,
            0.04,
            (
                "Matched-window mean spread "
                f"reduction: {reduction:.1f}%"
            ),
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            bbox={
                "facecolor": "white",
                "alpha": 0.8,
                "edgecolor": "0.7",
            },
        )

    ax.set_ylabel(
        "Mean pairwise track spread (km)"
    )
    ax.set_xlabel(
        "Valid time (UTC)"
    )
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend()

    ax.xaxis.set_major_formatter(
        mdates.DateFormatter("%d %b")
    )

    figure.autofmt_xdate()
    figure.tight_layout()

    figure.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(figure)


def plot_cycle_track_map(
    baseline: dict[str, dict],
    later: dict[str, dict],
    output_path: Path,
    title: str,
    baseline_label: str,
    later_label: str,
) -> None:
    """Plot baseline and later-cycle tracks using common model colors."""
    geographic_crs = ccrs.PlateCarree()

    first_item = next(
        iter(later.values())
    )

    seed = first_item[
        "provenance"
    ]["seed"]

    reference_longitude = (
        TRACK_MODULE.normalize_longitude(
            seed["longitude"]
        )
    )

    map_crs = ccrs.PlateCarree(
        central_longitude=(
            180.0
            if abs(reference_longitude) > 150.0
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

    common_models = sorted(
        set(baseline) & set(later)
    )

    colors = plt.rcParams[
        "axes.prop_cycle"
    ].by_key()["color"]

    for index, model in enumerate(
        common_models
    ):
        color = colors[
            index % len(colors)
        ]

        for cycle_name, tracks, linestyle in [
            (baseline_label, baseline, "--"),
            (later_label, later, "-"),
        ]:
            dataframe = tracks[
                model
            ]["dataframe"].sort_values(
                "valid_time"
            )

            longitude = (
                TRACK_MODULE.longitude_near_reference(
                    np.asarray(
                        [
                            TRACK_MODULE.
                            normalize_longitude(value)
                            for value in dataframe[
                                "longitude"
                            ]
                        ],
                        dtype=float,
                    ),
                    reference_longitude,
                )
            )

            latitude = dataframe[
                "latitude"
            ].to_numpy(
                dtype=float
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
                linestyle=linestyle,
                linewidth=1.7,
                color=color,
                label=(
                    f"{model_display_name(model)} — {cycle_name}"
                ),
                transform=geographic_crs,
            )

            ax.scatter(
                longitude[0],
                latitude[0],
                marker="s",
                s=35,
                color=color,
                transform=geographic_crs,
                zorder=5,
            )

            ax.scatter(
                longitude[-1],
                latitude[-1],
                marker="X",
                s=45,
                color=color,
                transform=geographic_crs,
                zorder=5,
            )

    baseline_seed = next(
        iter(baseline.values())
    )["provenance"]["seed"]

    later_seed = next(
        iter(later.values())
    )["provenance"]["seed"]

    for label, seed_item, marker in [
        (
            f"{baseline_label} NHC seed",
            baseline_seed,
            "*",
        ),
        (
            f"{later_label} NHC seed",
            later_seed,
            "P",
        ),
    ]:
        longitude = (
            TRACK_MODULE.longitude_near_reference(
                TRACK_MODULE.normalize_longitude(
                    seed_item["longitude"]
                ),
                reference_longitude,
            )
        )

        latitude = float(
            seed_item["latitude"]
        )

        ax.scatter(
            longitude,
            latitude,
            marker=marker,
            s=130,
            label=label,
            transform=geographic_crs,
            zorder=10,
        )

        all_longitudes.append(longitude)
        all_latitudes.append(latitude)

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

    ax.add_feature(
        cfeature.LAND,
        alpha=0.25,
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

    if abs(reference_longitude) > 150.0:
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

    ax.grid(alpha=0.3)

    ax.set_title(title)
    ax.legend(
        fontsize=8,
        ncol=2,
    )

    figure.tight_layout()

    figure.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(figure)

def main() -> None:
    args = build_parser().parse_args()

    baseline = TRACK_MODULE.discover_tracks(
        args.baseline_dir
    )
    later = TRACK_MODULE.discover_tracks(
        args.later_dir
    )

    validate_cycles(
        baseline,
        later,
    )

    baseline_provenance = next(
        iter(baseline.values())
    )["provenance"]

    later_provenance = next(
        iter(later.values())
    )["provenance"]

    baseline_init = baseline_provenance[
        "forecast"
    ]["initialization_time"]

    later_init = later_provenance[
        "forecast"
    ]["initialization_time"]

    storm = baseline_provenance["storm"]

    if args.output_dir is None:
        output_dir = (
            args.later_dir
            / "cycle_comparison"
        )
    else:
        output_dir = args.output_dir

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    displacement, displacement_summary = (
        build_cycle_displacement(
            baseline,
            later,
        )
    )

    spread_models = resolve_spread_models(
        baseline,
        later,
        args.spread_models,
    )

    baseline_spread = build_system_spread(
        baseline,
        "baseline",
        spread_models,
    )

    later_spread = build_system_spread(
        later,
        "later",
        spread_models,
    )

    spread = pd.concat(
        [
            baseline_spread,
            later_spread,
        ],
        ignore_index=True,
    )

    spread_summary = summarize_system_spread(
        spread
    )

    (
        baseline_matched,
        later_matched,
    ) = match_spread_valid_times(
        baseline_spread,
        later_spread,
    )

    matched_spread = pd.concat(
        [
            baseline_matched,
            later_matched,
        ],
        ignore_index=True,
    )

    matched_spread_summary = (
        summarize_system_spread(
            matched_spread
        )
    )

    displacement.to_csv(
        output_dir / "cycle_displacement.csv",
        index=False,
    )

    displacement_summary.to_csv(
        output_dir
        / "cycle_displacement_summary.csv",
        index=False,
    )

    spread.to_csv(
        output_dir / "system_spread.csv",
        index=False,
    )

    spread_summary.to_csv(
        output_dir
        / "system_spread_summary.csv",
        index=False,
    )

    matched_spread.to_csv(
        output_dir
        / "matched_system_spread.csv",
        index=False,
    )

    matched_spread_summary.to_csv(
        output_dir
        / "matched_system_spread_summary.csv",
        index=False,
    )

    base_title = (
        args.title
        if args.title is not None
        else (
            f"{storm['name']} — "
            f"{baseline_init} vs {later_init}"
        )
    )

    baseline_cycle_label = (
        pd.Timestamp(baseline_init).strftime("%d %b")
    )
    later_cycle_label = (
        pd.Timestamp(later_init).strftime("%d %b")
    )

    plot_cycle_track_map(
        baseline,
        later,
        output_dir / "cycle_track_map.png",
        (
            f"{base_title}\n"
            "Operational forecast-cycle track comparison"
        ),
        baseline_cycle_label,
        later_cycle_label,
    )

    plot_cycle_displacement(
        displacement,
        output_dir / "cycle_displacement.png",
        (
            f"{base_title}\n"
            "Cycle-to-cycle track displacement"
        ),
    )

    plot_matched_system_spread(
        matched_spread,
        output_dir / "matched_system_spread.png",
        (
            f"{base_title}\n"
            "Independent-system spread at matched valid times"
        ),
    )

    print("=" * 78)
    print(
        "AIWEATHER OPERATIONAL TC CYCLE COMPARISON"
    )
    print("=" * 78)
    print(
        f"Storm              : "
        f"{storm['id']} {storm['name']}"
    )
    print(
        f"Baseline init      : {baseline_init}"
    )
    print(
        f"Later init         : {later_init}"
    )
    print(
        "Common models      : "
        + ", ".join(
            sorted(
                set(baseline) & set(later)
            )
        )
    )
    print(
        "Spread systems     : "
        + ", ".join(
            spread_models
        )
    )
    print(
        f"Output             : {output_dir}"
    )

    print()
    print("CYCLE DISPLACEMENT")
    print("-" * 78)

    print(
        displacement_summary[
            [
                "model",
                "number_of_common_times",
                "mean_cycle_displacement_km",
                "maximum_cycle_displacement_km",
                "mean_delta_latitude_degrees",
                "final_delta_latitude_degrees",
            ]
        ].to_string(
            index=False
        )
    )

    print()
    print("SYSTEM SPREAD")
    print("-" * 78)

    print(
        spread_summary.to_string(
            index=False
        )
    )

    print()
    print("MATCHED-WINDOW SYSTEM SPREAD")
    print("-" * 78)

    print(
        matched_spread_summary.to_string(
            index=False
        )
    )

    print()
    print(
        "Note: cycle displacement and system "
        "spread are not verification errors."
    )


if __name__ == "__main__":
    main()
