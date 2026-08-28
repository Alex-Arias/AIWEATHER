#!/usr/bin/env python3
"""
Run an AIWeather deterministic forecast through the common runner API.

Examples
--------
Pangu3 / Fausto:

    python scripts/run_aiweather_forecast.py \
        pangu3 \
        20260719T000000 \
        --datasource gfs \
        --lead-time 240 \
        --device cuda

Pangu6 / Genevieve:

    python scripts/run_aiweather_forecast.py \
        pangu6 \
        20260724T000000 \
        --datasource gfs \
        --lead-time 240 \
        --device cuda

GraphCast example:

    python scripts/run_aiweather_forecast.py \
        graphcast \
        20260724T000000 \
        --datasource gfs \
        --lead-time 240 \
        --device cuda
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

from aiweather.forecast import ForecastRequest
from aiweather.runners import create_runner


def normalize_init_time(
    value: str,
) -> tuple[str, str]:
    """
    Normalize an initialization time.

    Accepted formats:
        YYYYMMDDTHHMMSS
        YYYY-MM-DDTHH:MM:SS

    Returns
    -------
    request_time
        ISO-like string used by ForecastRequest.
    path_time
        Compact timestamp used in canonical output paths.
    """

    formats = [
        "%Y%m%dT%H%M%S",
        "%Y-%m-%dT%H:%M:%S",
    ]

    parsed = None

    for fmt in formats:
        try:
            parsed = datetime.strptime(
                value,
                fmt,
            )
            break
        except ValueError:
            continue

    if parsed is None:
        raise ValueError(
            "Initialization time must use "
            "YYYYMMDDTHHMMSS or "
            "YYYY-MM-DDTHH:MM:SS."
        )

    return (
        parsed.strftime(
            "%Y-%m-%dT%H:%M:%S"
        ),
        parsed.strftime(
            "%Y%m%dT%H%M%S"
        ),
    )


def main() -> None:

    parser = argparse.ArgumentParser(
        description=(
            "Run an AIWeather deterministic forecast "
            "through the common runner API."
        )
    )

    parser.add_argument(
        "model",
        help=(
            "AIWeather model name, for example "
            "graphcast, aifs2, pangu3, or pangu6."
        ),
    )

    parser.add_argument(
        "init_time",
        help=(
            "Initialization time as YYYYMMDDTHHMMSS "
            "or YYYY-MM-DDTHH:MM:SS."
        ),
    )

    parser.add_argument(
        "--datasource",
        required=True,
        help=(
            "Datasource configured for the model, "
            "for example gfs or ifs."
        ),
    )

    parser.add_argument(
        "--datasource-source",
        default=None,
        help=(
            "Optional datasource backend or mirror, "
            "for example aws, azure, or ecmwf. "
            "If omitted, use the datasource default."
        ),
    )

    parser.add_argument(
        "--lead-time",
        type=int,
        default=240,
        help=(
            "Forecast horizon in hours. "
            "Default: 240."
        ),
    )

    parser.add_argument(
        "--device",
        default="cuda",
        choices=[
            "cpu",
            "cuda",
        ],
        help=(
            "Execution device. Default: cuda."
        ),
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Optional explicit forecast.zarr path. "
            "If omitted, use "
            "outputs/<model>/<init>/forecast.zarr."
        ),
    )

    args = parser.parse_args()

    request_time, path_time = (
        normalize_init_time(
            args.init_time
        )
    )

    if args.output is None:
        output_path = (
            Path("outputs")
            / args.model.lower()
            / path_time
            / "forecast.zarr"
        )
    else:
        output_path = args.output

    request = ForecastRequest(
        model=args.model,
        datasource=args.datasource,
        datasource_source=args.datasource_source,
        init_time=request_time,
        lead_time=args.lead_time,
        output_path=str(
            output_path
        ),
        device=args.device,
    )

    runner = create_runner(
        request.model
    )

    print("=" * 76)
    print(
        "AIWEATHER DETERMINISTIC FORECAST"
    )
    print("=" * 76)

    print()
    print(
        "Model               :",
        request.model,
    )
    print(
        "Datasource          :",
        request.datasource,
    )
    print(
        "Datasource source   :",
        (
            request.datasource_source
            if request.datasource_source is not None
            else "default"
        ),
    )
    print(
        "Initialization      :",
        request.init_time,
    )
    print(
        "Forecast horizon h  :",
        request.lead_time,
    )
    print(
        "Native timestep h   :",
        runner.MODEL_TIMESTEP,
    )
    print(
        "Forecast steps      :",
        (
            request.lead_time
            // runner.MODEL_TIMESTEP
        ),
    )
    print(
        "Device              :",
        request.device,
    )
    print(
        "Output              :",
        request.output_path,
    )

    print()

    runner.run(
        request
    )

    print()
    print("=" * 76)
    print("FORECAST COMPLETE")
    print("=" * 76)
    print()
    print(
        "Output:",
        request.output_path,
    )


if __name__ == "__main__":
    main()
