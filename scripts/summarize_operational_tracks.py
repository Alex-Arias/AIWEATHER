#!/usr/bin/env python3

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from aiweather.forecast import open_forecast


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Summarize exported operational tropical cyclone "
            "tracks and their provenance metadata."
        )
    )

    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help=(
            "Directory containing *_track.csv and "
            "*_track.provenance.json files."
        ),
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Output CSV path. Defaults to "
            "<input-dir>/track_summary.csv."
        ),
    )

    return parser


def forecast_horizon_hours(forecast_path: Path) -> int:
    forecast = open_forecast(forecast_path)

    lead_time = forecast.dataset["lead_time"]

    return int(
        lead_time.max().values
        / np.timedelta64(1, "h")
    )


def summarize_track(
    provenance_path: Path,
) -> dict:
    with provenance_path.open(
        encoding="utf-8"
    ) as handle:
        provenance = json.load(handle)

    forecast_info = provenance["forecast"]
    tracking_info = provenance["tracking"]
    seed_info = provenance["seed"]
    storm_info = provenance["storm"]
    software_info = provenance.get(
        "software",
        {},
    )

    forecast_path = Path(
        forecast_info["path"]
    )

    horizon_hours = forecast_horizon_hours(
        forecast_path
    )

    track_csv = provenance_path.with_name(
        provenance_path.name.replace(
            ".provenance.json",
            ".csv",
        )
    )

    dataframe = pd.read_csv(track_csv)

    if dataframe.empty:
        first_lead_time_hours = None
        last_lead_time_hours = None

        first_valid_time = None
        last_valid_time = None

        last_latitude = None
        last_longitude = None

        reaches_forecast_horizon = False

    else:
        first_row = dataframe.iloc[0]
        last_row = dataframe.iloc[-1]

        first_lead_time_hours = int(
            first_row["lead_time_hours"]
        )

        last_lead_time_hours = int(
            last_row["lead_time_hours"]
        )

        first_valid_time = str(
            first_row["valid_time"]
        )

        last_valid_time = str(
            last_row["valid_time"]
        )

        last_latitude = float(
            last_row["latitude"]
        )

        last_longitude = float(
            last_row["longitude"]
        )

        reaches_forecast_horizon = (
            last_lead_time_hours
            == horizon_hours
        )

    return {
        "storm_id": storm_info["id"],
        "storm_name": storm_info["name"],
        "model": forecast_info["model_name"],
        "forecast_id": forecast_info[
            "forecast_id"
        ],
        "forecast_path": forecast_info["path"],
        "datasource": forecast_info.get(
            "datasource"
        ),
        "datasource_source": (
            forecast_info.get(
                "datasource_source"
            )
        ),
        "initialization_time": (
            forecast_info[
                "initialization_time"
            ]
        ),
        "experiment_type": tracking_info[
            "experiment_type"
        ],
        "seed_latitude": seed_info[
            "latitude"
        ],
        "seed_longitude": seed_info[
            "longitude"
        ],
        "seed_valid_time": seed_info[
            "valid_time"
        ],
        "seed_source": seed_info["source"],
        "seed_source_issuance_time": (
            seed_info.get(
                "source_issuance_time"
            )
        ),
        "number_of_points": len(dataframe),
        "first_lead_time_hours": (
            first_lead_time_hours
        ),
        "last_lead_time_hours": (
            last_lead_time_hours
        ),
        "first_valid_time": first_valid_time,
        "last_valid_time": last_valid_time,
        "last_latitude": last_latitude,
        "last_longitude": last_longitude,
        "forecast_horizon_hours": (
            horizon_hours
        ),
        "reaches_forecast_horizon": (
            reaches_forecast_horizon
        ),
        "search_radius_km": tracking_info[
            "search_radius_km"
        ],
        "wind_radius_km": tracking_info[
            "wind_radius_km"
        ],
        "maximum_translation_speed_mps": (
            tracking_info[
                "maximum_translation_speed_mps"
            ]
        ),
        "git_commit": software_info.get(
            "git_commit"
        ),
    }


def main() -> None:
    args = build_parser().parse_args()

    input_dir = args.input_dir

    provenance_files = sorted(
        input_dir.glob(
            "*_track.provenance.json"
        )
    )

    if not provenance_files:
        raise FileNotFoundError(
            "No *_track.provenance.json files "
            f"found in {input_dir}"
        )

    rows = [
        summarize_track(path)
        for path in provenance_files
    ]

    dataframe = pd.DataFrame(rows)

    output_path = (
        args.output
        if args.output is not None
        else input_dir / "track_summary.csv"
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    dataframe.to_csv(
        output_path,
        index=False,
    )

    print("=" * 76)
    print("AIWEATHER OPERATIONAL TRACK SUMMARY")
    print("=" * 76)
    print()
    print("Input directory :", input_dir)
    print("Tracks found    :", len(dataframe))
    print("Output          :", output_path)
    print()

    display_columns = [
        "model",
        "number_of_points",
        "last_lead_time_hours",
        "forecast_horizon_hours",
        "reaches_forecast_horizon",
    ]

    print(
        dataframe[
            display_columns
        ].to_string(
            index=False
        )
    )


if __name__ == "__main__":
    main()
