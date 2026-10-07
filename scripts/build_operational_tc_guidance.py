"""Build standardized operational tropical-cyclone guidance.

This module consumes an existing AIWeather latest-cycle track-variability
product. It does not recompute tracks, consensus positions, or spread.

Track spread describes disagreement among the contributing forecast
members. It is not a calibrated probability of tropical-cyclone position.

Model availability is reported separately from spread magnitude.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


REQUIRED_COLUMNS = {
    "valid_time",
    "consensus_latitude",
    "consensus_longitude",
    "n_models",
    "r67_km",
    "r90_km",
    "max_distance_km",
    "max_lead_h",
    "spread_valid",
}


def availability_status(n_models: int) -> str:
    """Classify support from the number of contributing models."""
    if n_models >= 3:
        return "SUPPORTED"
    if n_models == 2:
        return "LIMITED"
    return "INSUFFICIENT"


def build_guidance(
    variability: pd.DataFrame,
    *,
    storm: str,
    tracker: str = "wuduan",
) -> pd.DataFrame:
    """Convert latest-cycle variability into operational TC guidance."""

    missing = REQUIRED_COLUMNS - set(variability.columns)
    if missing:
        raise ValueError(
            "Missing required columns: "
            + ", ".join(sorted(missing))
        )

    df = variability.copy()

    df["valid_time"] = pd.to_datetime(df["valid_time"])
    df = df.sort_values("valid_time").reset_index(drop=True)

    df["n_models"] = df["n_models"].astype(int)
    df["spread_valid"] = df["spread_valid"].astype(bool)

    df["availability_status"] = (
        df["n_models"].map(availability_status)
    )

    df["guidance_supported"] = (
        df["spread_valid"]
        & (df["n_models"] >= 3)
    )

    result = pd.DataFrame(
        {
            "storm": storm,
            "tracker": tracker,
            "valid_time": df["valid_time"],
            "lead_time_hours": df["max_lead_h"].astype(int),
            "consensus_latitude": df["consensus_latitude"],
            "consensus_longitude": df["consensus_longitude"],
            "n_models": df["n_models"],
            "r67_km": df["r67_km"],
            "r90_km": df["r90_km"],
            "max_distance_km": df["max_distance_km"],
            "spread_valid": df["spread_valid"],
            "availability_status": df["availability_status"],
            "guidance_supported": df["guidance_supported"],
        }
    )

    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build standardized operational TC guidance from an "
            "AIWeather latest-cycle variability CSV."
        )
    )

    parser.add_argument(
        "--variability",
        required=True,
        type=Path,
        help="Latest-cycle variability CSV.",
    )

    parser.add_argument(
        "--storm",
        required=True,
        help="Storm name.",
    )

    parser.add_argument(
        "--tracker",
        default="wuduan",
        help="Tracker name. Default: wuduan.",
    )

    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Output guidance CSV.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    variability = pd.read_csv(args.variability)

    guidance = build_guidance(
        variability,
        storm=args.storm,
        tracker=args.tracker,
    )

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    guidance.to_csv(
        args.output,
        index=False,
    )

    supported = int(guidance["guidance_supported"].sum())

    print("Operational TC guidance")
    print("-----------------------")
    print(f"Storm             : {args.storm}")
    print(f"Tracker           : {args.tracker}")
    print(f"Input             : {args.variability}")
    print(f"Output            : {args.output}")
    print(f"Guidance rows     : {len(guidance)}")
    print(
        "Supported rows    : "
        f"{supported}/{len(guidance)}"
    )
    print()
    print(
        "NOTE: spread describes multimodel forecast disagreement; "
        "it is not a calibrated probability of TC position."
    )


if __name__ == "__main__":
    main()
