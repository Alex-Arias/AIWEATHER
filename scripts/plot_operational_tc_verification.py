#!/usr/bin/env python3
"""Plot frozen operational TC verification at exact common valid times."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from aiweather.plotting.model_style import (
    model_color,
    model_label,
    model_linestyle,
    model_marker,
    model_sort_key,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot frozen operational tropical-cyclone "
            "verification at exact common valid times."
        )
    )

    parser.add_argument(
        "--input-dir",
        required=True,
        type=Path,
        help=(
            "Verification directory containing "
            "*_common_verification.csv files."
        ),
    )

    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="Directory for PNG and PDF figures.",
    )

    parser.add_argument(
        "--title",
        required=True,
        help="Figure title prefix.",
    )

    return parser.parse_args()


def load_common_verification(
    input_dir: Path,
) -> dict[str, pd.DataFrame]:

    paths = sorted(
        input_dir.glob(
            "*_common_verification.csv"
        )
    )

    if not paths:
        raise FileNotFoundError(
            "No *_common_verification.csv files "
            f"found in {input_dir}"
        )

    data = {}

    for path in paths:
        suffix = "_common_verification.csv"

        model = path.name[
            :-len(suffix)
        ]

        table = pd.read_csv(
            path
        )

        required = {
            "lead_time_hours",
            "valid_time",
            "track_error_km",
            "pressure_error_pa",
            "wind_error_ms",
        }

        missing = (
            required
            - set(table.columns)
        )

        if missing:
            raise ValueError(
                f"{path} is missing required columns: "
                + ", ".join(
                    sorted(missing)
                )
            )

        table = table.copy()

        table["valid_time"] = pd.to_datetime(
            table["valid_time"]
        )

        table = table.sort_values(
            "lead_time_hours"
        )

        data[model] = table

    return dict(
        sorted(
            data.items(),
            key=lambda item: model_sort_key(
                item[0]
            ),
        )
    )


def validate_common_times(
    data: dict[str, pd.DataFrame],
) -> None:

    reference_times = None
    reference_model = None

    for model, table in data.items():
        times = tuple(
            table["valid_time"]
        )

        if reference_times is None:
            reference_times = times
            reference_model = model
            continue

        if times != reference_times:
            raise ValueError(
                "Verification tables do not contain "
                "identical exact common valid times: "
                f"{reference_model} and {model} differ."
            )


def save_figure(
    fig,
    output_dir: Path,
    stem: str,
) -> tuple[Path, Path]:

    png = (
        output_dir
        / f"{stem}.png"
    )

    pdf = (
        output_dir
        / f"{stem}.pdf"
    )

    fig.savefig(
        png,
        dpi=300,
        bbox_inches="tight",
    )

    fig.savefig(
        pdf,
        bbox_inches="tight",
    )

    plt.close(
        fig
    )

    return png, pdf


def plot_track_error(
    data: dict[str, pd.DataFrame],
    output_dir: Path,
    title: str,
) -> tuple[Path, Path]:

    fig, ax = plt.subplots(
        figsize=(10, 6)
    )

    for model, table in data.items():

        ax.plot(
            table["lead_time_hours"],
            table["track_error_km"],
            marker=model_marker(
                model
            ),
            linestyle=model_linestyle(
                model
            ),
            markersize=5,
            linewidth=2,
            color=model_color(
                model
            ),
            label=model_label(
                model
            ),
        )

    ax.set_xlabel(
        "Forecast lead time (h)"
    )

    ax.set_ylabel(
        "Track error (km)"
    )

    ax.set_title(
        f"{title}\n"
        "Frozen Operational Forecast Verification — "
        "Exact Common Valid Times"
    )

    ax.grid(
        True,
        alpha=0.3,
    )

    ax.legend()

    fig.tight_layout()

    return save_figure(
        fig,
        output_dir,
        "track_error_common",
    )


def plot_pressure_error(
    data: dict[str, pd.DataFrame],
    output_dir: Path,
    title: str,
) -> tuple[Path, Path]:

    fig, ax = plt.subplots(
        figsize=(10, 6)
    )

    for model, table in data.items():

        pressure_error_hpa = (
            table["pressure_error_pa"]
            / 100.0
        )

        ax.plot(
            table["lead_time_hours"],
            pressure_error_hpa,
            marker=model_marker(
                model
            ),
            linestyle=model_linestyle(
                model
            ),
            markersize=5,
            linewidth=2,
            color=model_color(
                model
            ),
            label=model_label(
                model
            ),
        )

    ax.axhline(
        0.0,
        linewidth=1.0,
    )

    ax.set_xlabel(
        "Forecast lead time (h)"
    )

    ax.set_ylabel(
        "Central-pressure error (hPa)"
    )

    ax.set_title(
        f"{title}\n"
        "Forecast minus IBTrACS — Exact Common Valid Times"
    )

    ax.grid(
        True,
        alpha=0.3,
    )

    ax.legend()

    fig.tight_layout()

    return save_figure(
        fig,
        output_dir,
        "pressure_error_common",
    )


def plot_wind_error(
    data: dict[str, pd.DataFrame],
    output_dir: Path,
    title: str,
) -> tuple[Path, Path]:

    fig, ax = plt.subplots(
        figsize=(10, 6)
    )

    for model, table in data.items():

        ax.plot(
            table["lead_time_hours"],
            table["wind_error_ms"],
            marker=model_marker(
                model
            ),
            linestyle=model_linestyle(
                model
            ),
            markersize=5,
            linewidth=2,
            color=model_color(
                model
            ),
            label=model_label(
                model
            ),
        )

    ax.axhline(
        0.0,
        linewidth=1.0,
    )

    ax.set_xlabel(
        "Forecast lead time (h)"
    )

    ax.set_ylabel(
        "Maximum-wind error (m/s)"
    )

    ax.set_title(
        f"{title}\n"
        "Forecast minus IBTrACS — Exact Common Valid Times"
    )

    ax.grid(
        True,
        alpha=0.3,
    )

    ax.legend()

    fig.tight_layout()

    return save_figure(
        fig,
        output_dir,
        "wind_error_common",
    )


def main() -> None:
    args = parse_args()

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    data = load_common_verification(
        args.input_dir
    )

    validate_common_times(
        data
    )

    track_outputs = plot_track_error(
        data,
        args.output_dir,
        args.title,
    )

    pressure_outputs = plot_pressure_error(
        data,
        args.output_dir,
        args.title,
    )

    wind_outputs = plot_wind_error(
        data,
        args.output_dir,
        args.title,
    )

    first_table = next(
        iter(data.values())
    )

    first_lead = (
        first_table[
            "lead_time_hours"
        ].min()
    )

    last_lead = (
        first_table[
            "lead_time_hours"
        ].max()
    )

    print("=" * 78)
    print(
        "AIWEATHER FROZEN OPERATIONAL TC "
        "VERIFICATION PLOTS"
    )
    print("=" * 78)
    print(
        "Models          : "
        + ", ".join(data)
    )
    print(
        f"Common points   : {len(first_table)}"
    )
    print(
        f"Common leads    : "
        f"{first_lead:g} to {last_lead:g} h"
    )
    print(
        f"Output          : {args.output_dir}"
    )
    print()
    print(
        f"Track error     : {track_outputs[0]}"
    )
    print(
        f"Pressure error  : {pressure_outputs[0]}"
    )
    print(
        f"Wind error      : {wind_outputs[0]}"
    )


if __name__ == "__main__":
    main()
