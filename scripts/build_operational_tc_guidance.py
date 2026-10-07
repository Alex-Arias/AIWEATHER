"""Build standardized operational tropical-cyclone guidance.

This module consumes existing AIWeather latest-cycle track-variability
and lagged-member products. It does not recompute tracks, consensus
positions, or spread.

Track spread describes disagreement among contributing forecast
configurations. It is not a calibrated probability of tropical-cyclone
position.

Operational support is based on distinct model families rather than
raw forecast-configuration count. Pangu3 and Pangu6 are treated as
configurations of the same Pangu-Weather model family.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


REQUIRED_VARIABILITY_COLUMNS = {
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

REQUIRED_MEMBER_COLUMNS = {
    "valid_time",
    "cycle_time",
    "model",
}


def model_family(model: str) -> str:
    """Map operational configuration names to model families."""
    key = str(model).strip().lower()

    mapping = {
        "aifs2": "AIFS2",
        "graphcast": "GraphCast",
        "pangu3": "Pangu-Weather",
        "pangu6": "Pangu-Weather",
    }

    if key not in mapping:
        raise ValueError(
            f"Unknown operational model configuration: {model!r}"
        )

    return mapping[key]


def availability_status(n_model_families: int) -> str:
    """Classify support from distinct contributing model families."""
    if n_model_families >= 3:
        return "SUPPORTED"
    if n_model_families == 2:
        return "LIMITED"
    return "INSUFFICIENT"


def parse_bool_series(series: pd.Series) -> pd.Series:
    """Strictly parse boolean-like values."""

    if pd.api.types.is_bool_dtype(series):
        return series.astype(bool)

    mapping = {
        "true": True,
        "false": False,
        "1": True,
        "0": False,
    }

    normalized = (
        series.astype(str)
        .str.strip()
        .str.lower()
    )

    unknown = sorted(
        set(normalized.unique()) - set(mapping)
    )

    if unknown:
        raise ValueError(
            "Invalid boolean value(s) in spread_valid: "
            + ", ".join(repr(v) for v in unknown)
        )

    return normalized.map(mapping).astype(bool)


def build_guidance(
    variability: pd.DataFrame,
    members: pd.DataFrame,
    *,
    storm: str,
    tracker: str = "wuduan",
) -> pd.DataFrame:
    """Build model-family-aware latest-cycle TC guidance."""

    missing = (
        REQUIRED_VARIABILITY_COLUMNS
        - set(variability.columns)
    )

    if missing:
        raise ValueError(
            "Missing required variability columns: "
            + ", ".join(sorted(missing))
        )

    missing = (
        REQUIRED_MEMBER_COLUMNS
        - set(members.columns)
    )

    if missing:
        raise ValueError(
            "Missing required member columns: "
            + ", ".join(sorted(missing))
        )

    df = variability.copy()
    mem = members.copy()

    df["valid_time"] = pd.to_datetime(
        df["valid_time"]
    )

    mem["valid_time"] = pd.to_datetime(
        mem["valid_time"]
    )

    mem["cycle_time"] = pd.to_datetime(
        mem["cycle_time"]
    )

    df = df.sort_values(
        "valid_time"
    ).reset_index(drop=True)

    df["n_configurations"] = (
        pd.to_numeric(
            df["n_models"],
            errors="raise",
        ).astype(int)
    )

    df["spread_valid"] = parse_bool_series(
        df["spread_valid"]
    )

    latest_cycle = mem["cycle_time"].max()

    latest_members = mem[
        mem["cycle_time"] == latest_cycle
    ].copy()

    if latest_members.empty:
        raise ValueError(
            "No members found for latest operational cycle."
        )

    latest_members["model_family"] = (
        latest_members["model"].map(model_family)
    )

    inventory = (
        latest_members
        .groupby("valid_time")
        .agg(
            n_member_configurations=(
                "model",
                "nunique",
            ),
            n_model_families=(
                "model_family",
                "nunique",
            ),
        )
        .reset_index()
    )

    df = df.merge(
        inventory,
        on="valid_time",
        how="left",
        validate="one_to_one",
    )

    if df[
        [
            "n_member_configurations",
            "n_model_families",
        ]
    ].isna().any().any():
        raise ValueError(
            "Latest-cycle member inventory does not cover "
            "all variability valid times."
        )

    df["n_member_configurations"] = (
        df["n_member_configurations"].astype(int)
    )

    df["n_model_families"] = (
        df["n_model_families"].astype(int)
    )

    mismatch = (
        df["n_configurations"]
        != df["n_member_configurations"]
    )

    if mismatch.any():
        bad = df.loc[
            mismatch,
            [
                "valid_time",
                "n_configurations",
                "n_member_configurations",
            ],
        ]

        raise ValueError(
            "Variability/member configuration-count mismatch:\n"
            + bad.to_string(index=False)
        )

    df["availability_status"] = (
        df["n_model_families"]
        .map(availability_status)
    )

    df["guidance_supported"] = (
        df["spread_valid"]
        & (df["n_model_families"] >= 3)
    )

    return pd.DataFrame(
        {
            "storm": storm,
            "tracker": tracker,
            "valid_time": df["valid_time"],
            "lead_time_hours":
                df["max_lead_h"].astype(int),
            "consensus_latitude":
                df["consensus_latitude"],
            "consensus_longitude":
                df["consensus_longitude"],
            "n_configurations":
                df["n_configurations"],
            "n_model_families":
                df["n_model_families"],
            "r67_km": df["r67_km"],
            "r90_km": df["r90_km"],
            "max_distance_km":
                df["max_distance_km"],
            "spread_valid":
                df["spread_valid"],
            "availability_status":
                df["availability_status"],
            "guidance_supported":
                df["guidance_supported"],
        }
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build model-family-aware operational TC guidance "
            "from AIWeather latest-cycle variability and "
            "member products."
        )
    )

    parser.add_argument(
        "--variability",
        required=True,
        type=Path,
        help="Latest-cycle variability CSV.",
    )

    parser.add_argument(
        "--members",
        required=True,
        type=Path,
        help="Lagged-member CSV used to identify model families.",
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

    variability = pd.read_csv(
        args.variability
    )

    members = pd.read_csv(
        args.members
    )

    guidance = build_guidance(
        variability,
        members,
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

    supported = int(
        guidance["guidance_supported"].sum()
    )

    print("Operational TC guidance")
    print("-----------------------")
    print(f"Storm             : {args.storm}")
    print(f"Tracker           : {args.tracker}")
    print(f"Variability       : {args.variability}")
    print(f"Members           : {args.members}")
    print(f"Output            : {args.output}")
    print(f"Guidance rows     : {len(guidance)}")
    print(
        "Supported rows    : "
        f"{supported}/{len(guidance)}"
    )

    print()
    print(
        "NOTE: r67/r90 describe disagreement among forecast "
        "configurations; they are not calibrated probabilities "
        "of TC position."
    )

    print(
        "NOTE: operational support is based on distinct model "
        "families; Pangu3/Pangu6 count as one Pangu-Weather family."
    )


if __name__ == "__main__":
    main()
