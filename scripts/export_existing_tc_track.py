#!/usr/bin/env python3

import argparse
import subprocess
from datetime import datetime
from pathlib import Path

from aiweather.forecast import open_forecast
from aiweather.tracking import (
    build_existing_tc_track,
    export_operational_track,
)


def parse_datetime(value: str) -> datetime:
    """Parse an ISO-8601 datetime."""
    try:
        return datetime.fromisoformat(
            value.replace("Z", "+00:00")
        )
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"Invalid datetime: {value}"
        ) from exc


def get_git_commit() -> str | None:
    """Return the current Git commit when available."""
    try:
        result = subprocess.run(
            [
                "git",
                "rev-parse",
                "HEAD",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    except (
        FileNotFoundError,
        subprocess.CalledProcessError,
    ):
        return None

    return result.stdout.strip() or None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Track an existing tropical cyclone in an "
            "AIWeather forecast and export the track with "
            "reproducibility metadata."
        )
    )

    parser.add_argument(
        "--forecast",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--filename",
        required=True,
    )

    parser.add_argument(
        "--storm-id",
        required=True,
    )
    parser.add_argument(
        "--storm-name",
        required=True,
    )

    parser.add_argument(
        "--seed-latitude",
        type=float,
        required=True,
    )
    parser.add_argument(
        "--seed-longitude",
        type=float,
        required=True,
    )
    parser.add_argument(
        "--seed-valid-time",
        type=parse_datetime,
        required=True,
    )
    parser.add_argument(
        "--seed-source",
        required=True,
    )
    parser.add_argument(
        "--seed-source-issuance-time",
        type=parse_datetime,
    )

    parser.add_argument(
        "--experiment-type",
        required=True,
    )

    parser.add_argument(
        "--lat-min",
        type=float,
        required=True,
    )
    parser.add_argument(
        "--lat-max",
        type=float,
        required=True,
    )
    parser.add_argument(
        "--lon-min",
        type=float,
        required=True,
    )
    parser.add_argument(
        "--lon-max",
        type=float,
        required=True,
    )

    parser.add_argument(
        "--search-radius-km",
        type=float,
        default=500.0,
    )
    parser.add_argument(
        "--wind-radius-km",
        type=float,
        default=300.0,
    )
    parser.add_argument(
        "--maximum-translation-speed-mps",
        type=float,
        default=20.0,
    )

    parser.add_argument(
        "--datasource",
    )
    parser.add_argument(
        "--datasource-source",
    )

    return parser


def main() -> None:
    args = build_parser().parse_args()

    forecast = open_forecast(
        args.forecast
    )

    records = build_existing_tc_track(
        forecast,
        initial_latitude=args.seed_latitude,
        initial_longitude=args.seed_longitude,
        lat_min=args.lat_min,
        lat_max=args.lat_max,
        lon_min=args.lon_min,
        lon_max=args.lon_max,
        search_radius_km=args.search_radius_km,
        wind_radius_km=args.wind_radius_km,
        maximum_translation_speed_mps=(
            args.maximum_translation_speed_mps
        ),
    )

    csv_path, provenance_path = (
        export_operational_track(
            records,
            args.output_dir,
            filename=args.filename,
            forecast_path=args.forecast,
            forecast_metadata=forecast.metadata,
            storm_id=args.storm_id,
            storm_name=args.storm_name,
            seed_latitude=args.seed_latitude,
            seed_longitude=args.seed_longitude,
            seed_valid_time=args.seed_valid_time,
            seed_source=args.seed_source,
            seed_source_issuance_time=(
                args.seed_source_issuance_time
            ),
            experiment_type=args.experiment_type,
            lat_min=args.lat_min,
            lat_max=args.lat_max,
            lon_min=args.lon_min,
            lon_max=args.lon_max,
            search_radius_km=args.search_radius_km,
            wind_radius_km=args.wind_radius_km,
            maximum_translation_speed_mps=(
                args.maximum_translation_speed_mps
            ),
            datasource=args.datasource,
            datasource_source=args.datasource_source,
            git_commit=get_git_commit(),
        )
    )

    print("=" * 76)
    print("AIWEATHER EXISTING-TC TRACK EXPORT")
    print("=" * 76)
    print()
    print("Forecast            :", args.forecast)
    print("Model               :", forecast.metadata.model_name)
    print("Initialization      :", forecast.metadata.initialization_time)
    print("Storm               :", args.storm_name)
    print("Storm ID            :", args.storm_id)
    print(
        "Seed                :",
        f"{args.seed_latitude:.2f}, "
        f"{args.seed_longitude:.2f}",
    )
    print("Seed valid time     :", args.seed_valid_time)
    print("Track points        :", len(records))

    if records:
        print(
            "First lead h        :",
            records[0].lead_time_hours,
        )
        print(
            "Last lead h         :",
            records[-1].lead_time_hours,
        )
        print(
            "Last valid time     :",
            records[-1].valid_time,
        )
        print(
            "Last center         :",
            f"{records[-1].latitude:.2f}, "
            f"{records[-1].longitude:.2f}",
        )

    print()
    print("Track CSV           :", csv_path)
    print("Provenance JSON     :", provenance_path)


if __name__ == "__main__":
    main()
