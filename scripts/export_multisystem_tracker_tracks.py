#!/usr/bin/env python3
"""
Export WuDuan and Vitart multi-system tropical-cyclone tracks from an
AIWeather operational forecast cycle.

This workflow is forecast-only. It does not use IBTrACS, observed storm
positions, native AIWeather tracks, or one tracker to select the other.

For the current eastern Pacific two-system experiment, Earth2Studio
candidate paths are selected independently in western and eastern
geographic regions.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import pandas as pd

from aiweather.forecast import open_forecast
from aiweather.tracking import (
    earth2studio_track_to_records,
    run_vitart_tracker,
    run_wuduan_tracker,
)


# Broad tracker domain used by the operational experiment.
LAT_MIN = 5.0
LAT_MAX = 35.0
LON_MIN = -140.0
LON_MAX = -90.0

# Separate simultaneous systems without using observations.
WESTERN_REGION = {
    "lat_min": LAT_MIN,
    "lat_max": LAT_MAX,
    "lon_min": 220.0,
    "lon_max": 245.0,
}

EASTERN_REGION = {
    "lat_min": LAT_MIN,
    "lat_max": LAT_MAX,
    "lon_min": 245.0,
    "lon_max": 270.0,
}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Export operational WuDuan and Vitart tracks for "
            "simultaneous eastern-Pacific tropical cyclones."
        )
    )

    parser.add_argument(
        "--init",
        required=True,
        help="Initialization time as YYYYMMDDTHHMMSS.",
    )

    parser.add_argument(
        "--western-name",
        default="Western",
    )

    parser.add_argument(
        "--eastern-name",
        default="Eastern",
    )

    parser.add_argument(
        "--device",
        default="cuda",
        help="Torch device for Earth2Studio trackers.",
    )

    return parser.parse_args()


def git_commit():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True,
        ).strip()
    except Exception:
        return None


def get_forecasts(init):
    aifs2_default = Path(
        f"outputs/aifs2/{init}/forecast.zarr"
    )
    aifs2_legacy = Path(
        f"outputs/aifs2/{init}/forecast_azure.zarr"
    )

    if aifs2_default.exists():
        aifs2_path = aifs2_default
    elif aifs2_legacy.exists():
        aifs2_path = aifs2_legacy
    else:
        raise FileNotFoundError(
            "AIFS2 forecast not found. Checked: "
            f"{aifs2_default} and {aifs2_legacy}"
        )

    return {
        "graphcast": Path(
            f"outputs/graphcast/{init}/forecast.zarr"
        ),
        "aifs2": aifs2_path,
        "pangu3": Path(
            f"outputs/pangu3/{init}/forecast.zarr"
        ),
        "pangu6": Path(
            f"outputs/pangu6/{init}/forecast.zarr"
        ),
    }


def records_to_dataframe(records):
    return pd.DataFrame(
        [
            {
                "lead_time_hours":
                    record.lead_time_hours,
                "valid_time":
                    record.valid_time,
                "latitude":
                    record.latitude,
                "longitude":
                    record.longitude,
                "pressure":
                    record.pressure,
                "pressure_units":
                    record.pressure_units,
                "max_wind":
                    record.max_wind,
                "wind_units":
                    record.wind_units,
            }
            for record in records
        ]
    )


def select_originating_regional_track(
    tracks,
    *,
    lat_min,
    lat_max,
    lon_min,
    lon_max,
    excluded_path_ids=None,
):
    """
    Select the longest tracker path whose first point originates
    inside the requested geographic region.

    This prevents a long-lived track that originates in one region
    from being reassigned to another system merely because it later
    crosses the geographic classification boundary.

    Selection is forecast-only and does not use observations,
    IBTrACS, native AIWeather tracks, or another tracker.
    """
    excluded = set(
        excluded_path_ids or []
    )

    candidates = []

    for track in tracks:
        if len(track) == 0:
            continue

        if track.path_id in excluded:
            continue

        first_lat = float(
            track.latitude[0]
        )
        first_lon = float(
            track.longitude[0]
        )

        inside = (
            lat_min <= first_lat <= lat_max
            and lon_min <= first_lon <= lon_max
        )

        if not inside:
            continue

        candidates.append(track)

    if not candidates:
        return None

    # Require enough temporal support to distinguish a track
    # from an isolated or very short tracker detection.
    minimum_duration_hours = 12

    candidates = [
        track
        for track in candidates
        if (
            int(track.lead_time_hours[-1])
            - int(track.lead_time_hours[0])
        ) >= minimum_duration_hours
    ]

    if not candidates:
        return None

    # Prefer the candidate with the greatest temporal support.
    # Break ties using the number of tracker points.
    return max(
        candidates,
        key=lambda track: (
            int(track.lead_time_hours[-1])
            - int(track.lead_time_hours[0]),
            len(track),
        ),
    )


def main():
    args = parse_args()

    init = args.init

    forecasts = get_forecasts(init)

    systems = {
        args.western_name: (
            "western",
            WESTERN_REGION,
        ),
        args.eastern_name: (
            "eastern",
            EASTERN_REGION,
        ),
    }

    tracker_functions = {
        "wuduan": run_wuduan_tracker,
        "vitart": run_vitart_tracker,
    }

    commit = git_commit()

    summary_rows = []

    print("=" * 76)
    print(
        "AIWEATHER OPERATIONAL WUDUAN / VITART "
        "MULTI-SYSTEM EXPORT"
    )
    print("=" * 76)

    print("Initialization :", init)
    print("Device         :", args.device)
    print(
        "Western region :",
        WESTERN_REGION,
    )
    print(
        "Eastern region :",
        EASTERN_REGION,
    )
    print()

    for model, forecast_path in forecasts.items():

        print("=" * 76)
        print(model.upper())
        print("Forecast:", forecast_path)

        forecast = open_forecast(
            forecast_path
        )

        dataset = forecast.dataset

        initialization_time = (
            forecast.metadata.initialization_time
        )

        if initialization_time is None:
            raise ValueError(
                "Forecast initialization_time metadata "
                f"is missing for {forecast_path}."
            )

        for tracker_name, tracker_function in (
            tracker_functions.items()
        ):
            print()
            print("-" * 72)
            print(
                f"{model.upper()} / "
                f"{tracker_name.upper()}"
            )

            tracks = tracker_function(
                dataset,
                device=args.device,
            )

            print(
                "Candidate paths:",
                len(tracks),
            )

            for track in tracks:
                if len(track) == 0:
                    continue

                print(
                    f"  path={track.path_id:<3d} "
                    f"n={len(track):<3d} "
                    f"lead="
                    f"{int(track.lead_time_hours[0]):>3d}"
                    f"-"
                    f"{int(track.lead_time_hours[-1]):>3d} h "
                    f"first=("
                    f"{track.latitude[0]:.2f}, "
                    f"{track.longitude[0]:.2f})"
                )

            assigned_path_ids = set()

            for (
                storm_name,
                (
                    classification,
                    region,
                ),
            ) in systems.items():

                selected = select_originating_regional_track(
                    tracks,
                    lat_min=region["lat_min"],
                    lat_max=region["lat_max"],
                    lon_min=region["lon_min"],
                    lon_max=region["lon_max"],
                    excluded_path_ids=assigned_path_ids,
                )

                if selected is not None:
                    assigned_path_ids.add(
                        selected.path_id
                    )

                if selected is None:
                    print(
                        f"  {storm_name:<8s}: "
                        "NO TRACK"
                    )

                    # Remove stale outputs from an earlier run of
                    # this same cycle/storm/model/tracker. Without
                    # this cleanup, an old CSV could remain on disk
                    # even though the current detection is False.
                    output_dir = Path(
                        "results/operational"
                    ) / (
                        f"{storm_name.lower()}_{init}"
                    ) / model

                    stale_csv = output_dir / (
                        f"{model}_{tracker_name}_track.csv"
                    )
                    stale_provenance = output_dir / (
                        f"{model}_{tracker_name}_track."
                        "provenance.json"
                    )

                    removed = []

                    if stale_csv.exists():
                        stale_csv.unlink()
                        removed.append(str(stale_csv))

                    if stale_provenance.exists():
                        stale_provenance.unlink()
                        removed.append(
                            str(stale_provenance)
                        )

                    for stale_path in removed:
                        print(
                            "    removed stale:",
                            stale_path,
                        )

                    summary_rows.append(
                        {
                            "cycle": init,
                            "storm": storm_name,
                            "classification":
                                classification,
                            "model": model,
                            "tracker": tracker_name,
                            "detected": False,
                            "path_id": None,
                            "track_points": 0,
                            "first_lead_hours": None,
                            "last_lead_hours": None,
                        }
                    )

                    continue

                records = (
                    earth2studio_track_to_records(
                        selected,
                        initialization_time=(
                            initialization_time
                        ),
                    )
                )

                storm_slug = (
                    storm_name.lower()
                    .replace(" ", "_")
                )

                output_dir = (
                    Path("results")
                    / "operational"
                    / f"{storm_slug}_{init}"
                    / model
                )

                output_dir.mkdir(
                    parents=True,
                    exist_ok=True,
                )

                csv_path = (
                    output_dir
                    / f"{model}_{tracker_name}_track.csv"
                )

                provenance_path = (
                    output_dir
                    / (
                        f"{model}_{tracker_name}_track"
                        ".provenance.json"
                    )
                )

                records_to_dataframe(
                    records
                ).to_csv(
                    csv_path,
                    index=False,
                )

                provenance = {
                    "experiment_type":
                        "operational_multisystem_tracker",
                    "initialization_time":
                        str(initialization_time),
                    "forecast_path":
                        str(forecast_path),
                    "model":
                        model,
                    "tracker":
                        tracker_name,
                    "tracker_path_id":
                        int(selected.path_id),
                    "storm_name":
                        storm_name,
                    "storm_classification":
                        classification,
                    "selection_method":
                        "largest_number_of_points_in_region",
                    "selection_region":
                        region,
                    "uses_ibtracs_for_selection":
                        False,
                    "uses_observed_position_for_selection":
                        False,
                    "uses_native_track_for_selection":
                        False,
                    "device":
                        args.device,
                    "git_commit":
                        commit,
                }

                provenance_path.write_text(
                    json.dumps(
                        provenance,
                        indent=2,
                    )
                    + "\n"
                )

                first_lead = int(
                    selected.lead_time_hours[0]
                )
                last_lead = int(
                    selected.lead_time_hours[-1]
                )

                summary_rows.append(
                    {
                        "cycle": init,
                        "storm": storm_name,
                        "classification":
                            classification,
                        "model": model,
                        "tracker": tracker_name,
                        "detected": True,
                        "path_id":
                            int(selected.path_id),
                        "track_points":
                            len(selected),
                        "first_lead_hours":
                            first_lead,
                        "last_lead_hours":
                            last_lead,
                    }
                )

                print(
                    f"  {storm_name:<8s}: "
                    f"path={selected.path_id} "
                    f"n={len(selected)} "
                    f"lead={first_lead}-"
                    f"{last_lead} h"
                )

                print(
                    "    ->",
                    csv_path,
                )

    summary = pd.DataFrame(
        summary_rows
    )

    summary_path = (
        Path("results")
        / "operational"
        / (
            "tracker_detection_summary_"
            f"{init}.csv"
        )
    )

    summary.to_csv(
        summary_path,
        index=False,
    )

    print()
    print("=" * 76)
    print("SUMMARY")
    print("=" * 76)

    if not summary.empty:
        print(
            summary.to_string(
                index=False
            )
        )

    print()
    print("Summary:", summary_path)


if __name__ == "__main__":
    main()
