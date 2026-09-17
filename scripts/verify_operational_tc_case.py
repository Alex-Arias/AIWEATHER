#!/usr/bin/env python3
"""Verify frozen operational TC tracks against IBTrACS."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from aiweather.verification import verify_operational_track_csv
from aiweather.verification.lead_time import summarize_lead_time_verification


def _rmse(values: pd.Series) -> float:
    values = pd.to_numeric(values, errors="coerce")
    if values.notna().sum() == 0:
        return np.nan
    return float(np.sqrt(np.nanmean(np.square(values))))


def _summarize_table(
    table: pd.DataFrame,
) -> dict[str, float | int]:
    pressure_error = pd.to_numeric(
        table["pressure_error_pa"],
        errors="coerce",
    )
    wind_error = pd.to_numeric(
        table["wind_error_ms"],
        errors="coerce",
    )
    track_error = pd.to_numeric(
        table["track_error_km"],
        errors="coerce",
    )

    return {
        "overlap_count": int(len(table)),
        "first_lead_time_hours":
            table["lead_time_hours"].min(),
        "last_lead_time_hours":
            table["lead_time_hours"].max(),
        "mean_track_error_km":
            track_error.mean(),
        "rmse_track_error_km":
            _rmse(track_error),
        "median_track_error_km":
            track_error.median(),
        "maximum_track_error_km":
            track_error.max(),
        "pressure_bias_pa":
            pressure_error.mean(),
        "pressure_mae_pa":
            pressure_error.abs().mean(),
        "pressure_rmse_pa":
            _rmse(pressure_error),
        "wind_bias_ms":
            wind_error.mean(),
        "wind_mae_ms":
            wind_error.abs().mean(),
        "wind_rmse_ms":
            _rmse(wind_error),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Verify already-frozen operational tropical-cyclone "
            "tracks against IBTrACS."
        )
    )
    parser.add_argument(
        "--input-dir",
        required=True,
        type=Path,
        help="Operational case directory containing model subdirectories.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="Directory for verification outputs.",
    )
    parser.add_argument(
        "--ibtracs",
        required=True,
        type=Path,
        help="IBTrACS basin CSV.",
    )
    parser.add_argument(
        "--sid",
        required=True,
        help="IBTrACS storm SID.",
    )
    parser.add_argument(
        "--initialization-time",
        required=True,
        help="Forecast initialization time.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    track_paths = sorted(
        args.input_dir.glob("*/*_track.csv")
    )

    if not track_paths:
        raise FileNotFoundError(
            f"No frozen track CSV files found under {args.input_dir}"
        )

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary_rows = []
    lead_time_tables = []
    verified_tables = {}
    track_path_by_model = {}

    for track_path in track_paths:
        model = track_path.parent.name

        result = verify_operational_track_csv(
            track_path,
            args.ibtracs,
            sid=args.sid,
            initialization_time=args.initialization_time,
            forecast_name=model,
            observation_name="IBTrACS",
        )

        table = result.table.copy()
        verified_tables[model] = table
        track_path_by_model[model] = track_path

        table.to_csv(
            args.output_dir / f"{model}_verification.csv",
            index=False,
        )

        summary_rows.append(
            {
                "model": model,
                "forecast_track": str(track_path),
                "sid": args.sid,
                "initialization_time":
                    args.initialization_time,
                **_summarize_table(table),
            }
        )

        lead_summary = summarize_lead_time_verification(
            table
        )
        lead_summary.insert(
            0,
            "model",
            model,
        )
        lead_time_tables.append(
            lead_summary
        )

    summary = pd.DataFrame(
        summary_rows
    ).sort_values(
        "model"
    )

    summary.to_csv(
        args.output_dir / "verification_summary.csv",
        index=False,
    )

    lead_time_summary = pd.concat(
        lead_time_tables,
        ignore_index=True,
    )

    lead_time_summary.to_csv(
        args.output_dir / "lead_time_summary.csv",
        index=False,
    )

    common_times = None

    for table in verified_tables.values():
        times = set(
            pd.to_datetime(
                table["valid_time"]
            )
        )

        if common_times is None:
            common_times = times
        else:
            common_times &= times

    common_times = sorted(
        common_times or []
    )

    if not common_times:
        raise ValueError(
            "No exact common verification valid times "
            "were found across models."
        )

    common_rows = []
    common_lead_tables = []

    for model, table in verified_tables.items():
        valid_time = pd.to_datetime(
            table["valid_time"]
        )

        common_table = table[
            valid_time.isin(common_times)
        ].copy()

        common_table.to_csv(
            args.output_dir
            / f"{model}_common_verification.csv",
            index=False,
        )

        common_rows.append(
            {
                "model": model,
                "forecast_track":
                    str(track_path_by_model[model]),
                "sid": args.sid,
                "initialization_time":
                    args.initialization_time,
                **_summarize_table(common_table),
            }
        )

        common_lead = (
            summarize_lead_time_verification(
                common_table
            )
        )
        common_lead.insert(
            0,
            "model",
            model,
        )
        common_lead_tables.append(
            common_lead
        )

    common_summary = pd.DataFrame(
        common_rows
    ).sort_values(
        "model"
    )

    common_summary.to_csv(
        args.output_dir
        / "common_verification_summary.csv",
        index=False,
    )

    common_lead_summary = pd.concat(
        common_lead_tables,
        ignore_index=True,
    )

    common_lead_summary.to_csv(
        args.output_dir
        / "common_lead_time_summary.csv",
        index=False,
    )


    print("=" * 78)
    print("AIWEATHER FROZEN OPERATIONAL TC VERIFICATION")
    print("=" * 78)
    print(f"Input directory : {args.input_dir}")
    print(f"IBTrACS SID     : {args.sid}")
    print(f"Models verified : {len(summary)}")
    print(f"Output          : {args.output_dir}")
    print()
    print(
        summary[
            [
                "model",
                "overlap_count",
                "last_lead_time_hours",
                "mean_track_error_km",
                "rmse_track_error_km",
                "maximum_track_error_km",
            ]
        ].to_string(index=False)
    )
    print()
    print("COMMON VALID-TIME VERIFICATION")
    print("-" * 78)
    print(
        common_summary[
            [
                "model",
                "overlap_count",
                "last_lead_time_hours",
                "mean_track_error_km",
                "rmse_track_error_km",
                "maximum_track_error_km",
            ]
        ].to_string(index=False)
    )
    print()
    print(
        "Reference enters only after the operational tracks "
        "have been frozen."
    )


if __name__ == "__main__":
    main()
