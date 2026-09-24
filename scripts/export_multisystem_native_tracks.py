#!/usr/bin/env python3
"""
Export native multi-system tropical-cyclone tracks from an AIWeather
operational forecast cycle.

The exporter uses build_native_tc_tracks() exactly as the operational
multi-genesis plotting workflow does. No observed storm position,
IBTrACS best track, WuDuan track, or Vitart track is used to initialize
or select the native detections.

For the current eastern Pacific two-system experiment:
    longitude < -115 deg -> western system
    longitude >= -115 deg -> eastern system
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import pandas as pd

from aiweather.forecast import open_forecast
from aiweather.tracking import build_native_tc_tracks


LAT_MIN = 5.0
LAT_MAX = 35.0
LON_MIN = -140.0
LON_MAX = -90.0

CLASSIFICATION_LONGITUDE = -115.0
MAXIMUM_TRANSLATION_SPEED_MPS = 20.0


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Export native multi-system tropical-cyclone tracks "
            "for one operational AIWeather forecast cycle."
        )
    )

    parser.add_argument(
        "--init",
        required=True,
        help="Initialization time as YYYYMMDDTHHMMSS.",
    )

    parser.add_argument(
        "--western-name",
        default="Odalys",
    )

    parser.add_argument(
        "--eastern-name",
        default="Polo",
    )

    return parser.parse_args()


def lon180(longitude):
    longitude = float(longitude)
    return (
        longitude - 360.0
        if longitude > 180.0
        else longitude
    )


def git_commit():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True,
        ).strip()
    except Exception:
        return None


def forecast_paths(init):
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


def records_dataframe(records):
    return pd.DataFrame(
        [
            {
                "lead_time_hours":
                    record.lead_time_hours,
                "valid_time":
                    str(record.valid_time),
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



def select_regional_native_track(
    results,
    *,
    classification,
):
    """
    Select one native tropical-cyclone detection for an operational
    geographic classification.

    Candidates are classified from their genesis longitude. Selection
    follows the native genesis convention by preferring the earliest
    genesis lead and then the lowest genesis pressure. Additional
    deterministic tie-breakers favor stronger qualifying persistence,
    greater track support, and finally the lower native track index.

    This selection is forecast-only and does not use observations,
    IBTrACS, WuDuan, Vitart, or another forecast model.
    """
    if classification not in {
        "western",
        "eastern",
    }:
        raise ValueError(
            "classification must be 'western' or 'eastern'."
        )

    candidates = []

    for genesis, records in results:
        genesis_lon_180 = lon180(
            genesis.longitude
        )

        is_western = (
            genesis_lon_180
            < CLASSIFICATION_LONGITUDE
        )

        if classification == "western":
            inside = is_western
        else:
            inside = not is_western

        if inside:
            candidates.append(
                (genesis, records)
            )

    if not candidates:
        return None

    return min(
        candidates,
        key=lambda item: (
            int(
                item[0].genesis_lead_time_hours
            ),
            float(
                item[0].pressure
            ),
            -int(
                item[0].qualifying_points
            ),
            -len(
                item[1]
            ),
            int(
                item[0].track_index
            ),
        ),
    )

def main():
    args = parse_args()

    init = args.init

    names = {
        "western": args.western_name,
        "eastern": args.eastern_name,
    }

    commit = git_commit()

    summary_rows = []

    print("=" * 76)
    print("AIWEATHER NATIVE MULTI-SYSTEM TRACK EXPORT")
    print("=" * 76)
    print("Initialization :", init)
    print(
        "Domain         :",
        LAT_MIN,
        LAT_MAX,
        LON_MIN,
        LON_MAX,
    )
    print(
        "Classification : western <",
        CLASSIFICATION_LONGITUDE,
        "<= eastern",
    )
    print(
        "Max translation:",
        MAXIMUM_TRANSLATION_SPEED_MPS,
        "m/s",
    )
    print()

    for model, forecast_path in forecast_paths(init).items():

        print("-" * 76)
        print(model.upper())
        print("Forecast:", forecast_path)

        if not forecast_path.exists():
            raise FileNotFoundError(
                f"Forecast not found: {forecast_path}"
            )

        forecast = open_forecast(
            forecast_path
        )

        results = build_native_tc_tracks(
            forecast,
            lat_min=LAT_MIN,
            lat_max=LAT_MAX,
            lon_min=LON_MIN,
            lon_max=LON_MAX,
            maximum_translation_speed_mps=(
                MAXIMUM_TRANSLATION_SPEED_MPS
            ),
        )

        print("Detected systems:", len(results))

        for classification in (
            "western",
            "eastern",
        ):
            selected = select_regional_native_track(
                results,
                classification=classification,
            )

            if selected is None:
                print(
                    f"  {names[classification]:<8s} "
                    "no qualifying native detection"
                )
                continue

            genesis, records = selected

            genesis_lon_180 = lon180(
                genesis.longitude
            )

            storm_name = names[
                classification
            ]

            storm_slug = storm_name.lower()

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
                / f"{model}_track.csv"
            )

            provenance_path = (
                output_dir
                / f"{model}_track.provenance.json"
            )

            table = records_dataframe(
                records
            )

            table.to_csv(
                csv_path,
                index=False,
            )

            provenance = {
                "experiment_type":
                    "operational_native_detection",
                "initialization_time":
                    init,
                "storm_name":
                    storm_name,
                "storm_classification":
                    classification,
                "model":
                    model,
                "forecast_path":
                    str(forecast_path),
                "tracker":
                    "build_native_tc_tracks",
                "domain": {
                    "lat_min": LAT_MIN,
                    "lat_max": LAT_MAX,
                    "lon_min": LON_MIN,
                    "lon_max": LON_MAX,
                },
                "classification_longitude":
                    CLASSIFICATION_LONGITUDE,
                "maximum_translation_speed_mps":
                    MAXIMUM_TRANSLATION_SPEED_MPS,
                "native_detection": {
                    "track_index":
                        genesis.track_index,
                    "lead_time_hours":
                        genesis.genesis_lead_time_hours,
                    "latitude":
                        genesis.latitude,
                    "longitude":
                        genesis.longitude,
                    "longitude_180":
                        genesis_lon_180,
                    "pressure_pa":
                        genesis.pressure,
                    "max_wind_ms":
                        genesis.max_wind,
                    "qualifying_points":
                        genesis.qualifying_points,
                },
                "number_of_track_records":
                    len(records),
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

            last_lead = (
                records[-1].lead_time_hours
                if records
                else None
            )

            summary_rows.append(
                {
                    "storm": storm_name,
                    "model": model,
                    "genesis_lead_hours":
                        genesis.genesis_lead_time_hours,
                    "genesis_latitude":
                        genesis.latitude,
                    "genesis_longitude":
                        genesis_lon_180,
                    "genesis_pressure_pa":
                        genesis.pressure,
                    "genesis_max_wind_ms":
                        genesis.max_wind,
                    "track_points":
                        len(records),
                    "last_lead_hours":
                        last_lead,
                }
            )

            print(
                f"  {storm_name:<8s} "
                f"gen={genesis.genesis_lead_time_hours:3d} h  "
                f"lat={genesis.latitude:6.2f}  "
                f"lon={genesis_lon_180:7.2f}  "
                f"n={len(records):2d}  "
                f"last={last_lead}"
            )
            print(
                "    ->",
                csv_path,
            )

    summary = pd.DataFrame(
        summary_rows
    ).sort_values(
        [
            "storm",
            "model",
        ]
    )

    summary_path = (
        Path("results")
        / "operational"
        / f"native_detection_summary_{init}.csv"
    )

    summary.to_csv(
        summary_path,
        index=False,
    )

    print()
    print("=" * 76)
    print("SUMMARY")
    print("=" * 76)
    print(
        summary.to_string(
            index=False
        )
    )
    print()
    print("Summary:", summary_path)


if __name__ == "__main__":
    main()
