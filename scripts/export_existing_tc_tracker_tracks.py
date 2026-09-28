#!/usr/bin/env python3
"""
Export WuDuan and Vitart tracks for one existing tropical cyclone.

Cycle 1 associates Earth2Studio tracker candidates with the already
frozen seeded Native AIWeather track for the named storm.

Later cycles may instead use the previous operational tracker track
for storm-continuity association.

The workflow is forecast-only after the Native reference has been
frozen. IBTrACS is not used for tracker selection.
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
    read_track_records_csv,
    run_vitart_tracker,
    run_wuduan_tracker,
    select_matching_track,
)


MODELS = (
    "aifs2",
    "graphcast",
    "pangu3",
    "pangu6",
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Export operational WuDuan and Vitart tracks for "
            "one existing tropical cyclone."
        )
    )

    parser.add_argument(
        "--init",
        required=True,
        help="Initialization time as YYYYMMDDTHHMMSS.",
    )

    parser.add_argument(
        "--storm-name",
        required=True,
    )

    parser.add_argument(
        "--reference-init",
        default=None,
        help=(
            "Previous operational initialization used for "
            "tracker continuity. If omitted, each tracker is "
            "matched to the current-cycle frozen Native track."
        ),
    )

    parser.add_argument(
        "--association-minimum-overlap",
        type=int,
        default=2,
    )

    parser.add_argument(
        "--association-maximum-mean-error-km",
        type=float,
        default=600.0,
    )

    parser.add_argument(
        "--device",
        default="cuda",
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

    paths = {
        "aifs2": aifs2_path,
        "graphcast": Path(
            f"outputs/graphcast/{init}/forecast.zarr"
        ),
        "pangu3": Path(
            f"outputs/pangu3/{init}/forecast.zarr"
        ),
        "pangu6": Path(
            f"outputs/pangu6/{init}/forecast.zarr"
        ),
    }

    missing = [
        str(path)
        for path in paths.values()
        if not path.exists()
    ]

    if missing:
        raise FileNotFoundError(
            "Forecast(s) not found:\n"
            + "\n".join(missing)
        )

    return paths


def storm_slug(name):
    return (
        name.lower()
        .replace(" ", "_")
    )


def native_reference_path(
    *,
    storm_name,
    init,
    model,
):
    slug = storm_slug(storm_name)

    return (
        Path("results")
        / "operational"
        / f"{slug}_{init}"
        / model
        / f"{model}_track.csv"
    )


def previous_tracker_reference_path(
    *,
    storm_name,
    init,
    model,
    tracker_name,
):
    slug = storm_slug(storm_name)

    return (
        Path("results")
        / "operational"
        / f"{slug}_{init}"
        / model
        / f"{model}_{tracker_name}_track.csv"
    )


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


def select_existing_storm_track(
    reference_records,
    tracks,
    *,
    initialization_time,
    minimum_overlap=2,
    maximum_mean_error_km=600.0,
):
    """
    Select the tracker candidate matching an existing-storm reference.

    Matching is performed at exact common valid times.
    """
    return select_matching_track(
        reference_records,
        tracks,
        initialization_time=initialization_time,
        minimum_overlap=minimum_overlap,
        maximum_mean_error_km=maximum_mean_error_km,
        reference_name="existing_storm_reference",
        candidate_name="tracker_candidate",
    )


def remove_stale_products(
    *,
    storm_name,
    init,
    model,
    tracker_name,
):
    output_dir = (
        Path("results")
        / "operational"
        / f"{storm_slug(storm_name)}_{init}"
        / model
    )

    paths = [
        output_dir
        / f"{model}_{tracker_name}_track.csv",
        output_dir
        / (
            f"{model}_{tracker_name}_track"
            ".provenance.json"
        ),
    ]

    removed = []

    for path in paths:
        if path.exists():
            path.unlink()
            removed.append(path)

    return removed


def main():
    args = parse_args()

    if args.association_minimum_overlap < 1:
        raise ValueError(
            "--association-minimum-overlap must be >= 1."
        )

    if args.association_maximum_mean_error_km <= 0.0:
        raise ValueError(
            "--association-maximum-mean-error-km "
            "must be > 0."
        )

    init = args.init
    forecasts = get_forecasts(init)

    tracker_functions = {
        "wuduan": run_wuduan_tracker,
        "vitart": run_vitart_tracker,
    }

    commit = git_commit()
    summary_rows = []

    print("=" * 76)
    print(
        "AIWEATHER OPERATIONAL EXISTING-TC "
        "WUDUAN / VITART EXPORT"
    )
    print("=" * 76)
    print("Storm          :", args.storm_name)
    print("Initialization :", init)
    print("Device         :", args.device)
    print(
        "Reference      :",
        (
            "current-cycle frozen Native track"
            if args.reference_init is None
            else (
                "previous-cycle tracker continuity "
                f"({args.reference_init})"
            )
        ),
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

        for (
            tracker_name,
            tracker_function,
        ) in tracker_functions.items():

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
                    "-"
                    f"{int(track.lead_time_hours[-1]):>3d} h "
                    f"first=("
                    f"{track.latitude[0]:.2f}, "
                    f"{track.longitude[0]:.2f})"
                )

            if args.reference_init is None:
                reference_path = (
                    native_reference_path(
                        storm_name=args.storm_name,
                        init=init,
                        model=model,
                    )
                )

                association_method = (
                    "current_cycle_seeded_native_reference"
                )
            else:
                reference_path = (
                    previous_tracker_reference_path(
                        storm_name=args.storm_name,
                        init=args.reference_init,
                        model=model,
                        tracker_name=tracker_name,
                    )
                )

                association_method = (
                    "previous_cycle_valid_time_continuity"
                )

            if not reference_path.is_file():
                raise FileNotFoundError(
                    "Reference track not found: "
                    f"{reference_path}"
                )

            reference_records = (
                read_track_records_csv(
                    reference_path
                )
            )

            match = select_existing_storm_track(
                reference_records,
                tracks,
                initialization_time=(
                    initialization_time
                ),
                minimum_overlap=(
                    args.association_minimum_overlap
                ),
                maximum_mean_error_km=(
                    args
                    .association_maximum_mean_error_km
                ),
            )

            if match is None:
                print(
                    f"  {args.storm_name}: NO MATCH"
                )

                removed = remove_stale_products(
                    storm_name=args.storm_name,
                    init=init,
                    model=model,
                    tracker_name=tracker_name,
                )

                for path in removed:
                    print(
                        "    removed stale:",
                        path,
                    )

                summary_rows.append(
                    {
                        "cycle": init,
                        "storm": args.storm_name,
                        "model": model,
                        "tracker": tracker_name,
                        "detected": False,
                        "path_id": None,
                        "track_points": 0,
                        "first_lead_hours": None,
                        "last_lead_hours": None,
                        "overlap_count": None,
                        "mean_track_error_km": None,
                    }
                )

                continue

            selected = match.track

            records = (
                earth2studio_track_to_records(
                    selected,
                    initialization_time=(
                        initialization_time
                    ),
                )
            )

            output_dir = (
                Path("results")
                / "operational"
                / (
                    f"{storm_slug(args.storm_name)}_"
                    f"{init}"
                )
                / model
            )

            output_dir.mkdir(
                parents=True,
                exist_ok=True,
            )

            csv_path = (
                output_dir
                / (
                    f"{model}_{tracker_name}_"
                    "track.csv"
                )
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
                    "operational_existing_tc_tracker",
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
                    args.storm_name,
                "selection_method":
                    association_method,
                "reference_init":
                    args.reference_init,
                "reference_path":
                    str(reference_path),
                "storm_association": {
                    "overlap_count":
                        match.overlap_count,
                    "mean_track_error_km":
                        match.mean_track_error_km,
                    "minimum_overlap":
                        args.association_minimum_overlap,
                    "maximum_mean_error_km":
                        (
                            args
                            .association_maximum_mean_error_km
                        ),
                },
                "uses_ibtracs_for_selection":
                    False,
                "uses_observed_position_for_selection":
                    False,
                "uses_native_track_for_selection":
                    args.reference_init is None,
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
                    "storm": args.storm_name,
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
                    "overlap_count":
                        match.overlap_count,
                    "mean_track_error_km":
                        match.mean_track_error_km,
                }
            )

            print(
                f"  MATCH path={selected.path_id} "
                f"n={len(selected)} "
                f"lead={first_lead}-{last_lead} h"
            )
            print(
                "    overlap:",
                match.overlap_count,
            )
            print(
                "    mean separation km:",
                f"{match.mean_track_error_km:.2f}",
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
            f"{storm_slug(args.storm_name)}_"
            f"tracker_detection_summary_{init}.csv"
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
