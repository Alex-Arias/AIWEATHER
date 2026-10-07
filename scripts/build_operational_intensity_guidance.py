"""Build operational tropical-cyclone intensity guidance.

The guidance is derived from existing WuDuan operational track products
for three AIWeather model families:

    AIFS2
    GraphCast
    Pangu-Weather (6-h)

The module does not bias-correct forecast intensity and does not interpret
small intermodel spread as calibrated intensity confidence.

Pressure is standardized to hPa and maximum wind to m/s.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


REQUIRED_COLUMNS = {
    "lead_time_hours",
    "valid_time",
    "pressure",
    "pressure_units",
    "max_wind",
    "wind_units",
}


MODEL_COLUMNS = {
    "aifs2": "AIFS2",
    "graphcast": "GraphCast",
    "pangu": "Pangu-Weather (6-h)",
}


def availability_status(n_models: int) -> str:
    """Describe model-family availability."""
    if n_models >= 3:
        return "SUPPORTED"
    if n_models == 2:
        return "LIMITED"
    return "INSUFFICIENT"


def _read_track(
    path: Path,
    *,
    model_key: str,
) -> pd.DataFrame:
    """Read and standardize one WuDuan intensity track."""

    df = pd.read_csv(path)

    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(
            f"{path}: missing required columns: "
            + ", ".join(sorted(missing))
        )

    df = df.copy()

    df["valid_time"] = pd.to_datetime(df["valid_time"])
    df["lead_time_hours"] = pd.to_numeric(
        df["lead_time_hours"],
        errors="raise",
    ).astype(int)

    # Advisory interface uses a common 6-hour cadence.
    df = df[
        df["lead_time_hours"] % 6 == 0
    ].copy()

    pressure_units = set(
        df["pressure_units"]
        .dropna()
        .astype(str)
    )

    wind_units = set(
        df["wind_units"]
        .dropna()
        .astype(str)
    )

    if pressure_units != {"Pa"}:
        raise ValueError(
            f"{path}: expected pressure units Pa; "
            f"found {sorted(pressure_units)}"
        )

    if wind_units != {"m/s"}:
        raise ValueError(
            f"{path}: expected wind units m/s; "
            f"found {sorted(wind_units)}"
        )

    out = df[
        [
            "valid_time",
            "lead_time_hours",
            "pressure",
            "max_wind",
        ]
    ].copy()

    out = out.rename(
        columns={
            "lead_time_hours":
                f"{model_key}_lead_time_hours",
            "pressure":
                f"{model_key}_pressure_pa",
            "max_wind":
                f"{model_key}_wind_ms",
        }
    )

    out[f"{model_key}_pressure_hpa"] = (
        out.pop(f"{model_key}_pressure_pa")
        / 100.0
    )

    return out


def build_intensity_guidance(
    *,
    aifs2_path: Path,
    graphcast_path: Path,
    pangu_path: Path,
    storm: str,
    tracker: str = "wuduan",
) -> pd.DataFrame:
    """Build descriptive multimodel operational intensity guidance."""

    paths = {
        "aifs2": aifs2_path,
        "graphcast": graphcast_path,
        "pangu": pangu_path,
    }

    frames = {
        key: _read_track(
            Path(path),
            model_key=key,
        )
        for key, path in paths.items()
    }

    # Outer merge is deliberate: retain times at which one family
    # has stopped contributing so availability degradation remains
    # visible in the operational product.
    merged = None

    for frame in frames.values():
        if merged is None:
            merged = frame
        else:
            merged = merged.merge(
                frame,
                on="valid_time",
                how="outer",
                validate="one_to_one",
            )

    merged = merged.sort_values(
        "valid_time"
    ).reset_index(drop=True)

    pressure_cols = [
        "aifs2_pressure_hpa",
        "graphcast_pressure_hpa",
        "pangu_pressure_hpa",
    ]

    wind_cols = [
        "aifs2_wind_ms",
        "graphcast_wind_ms",
        "pangu_wind_ms",
    ]

    lead_cols = [
        "aifs2_lead_time_hours",
        "graphcast_lead_time_hours",
        "pangu_lead_time_hours",
    ]

    merged["n_models"] = (
        merged[pressure_cols]
        .notna()
        .sum(axis=1)
    )

    # A valid contributing model must contain both intensity fields.
    complete_models = pd.DataFrame(
        {
            key: (
                merged[f"{key}_pressure_hpa"].notna()
                & merged[f"{key}_wind_ms"].notna()
            )
            for key in MODEL_COLUMNS
        }
    )

    merged["n_models"] = complete_models.sum(axis=1)

    merged["availability_status"] = (
        merged["n_models"]
        .astype(int)
        .map(availability_status)
    )

    merged["intensity_supported"] = (
        merged["n_models"] >= 3
    )

    merged["pressure_mean_hpa"] = (
        merged[pressure_cols].mean(axis=1)
    )

    merged["pressure_min_hpa"] = (
        merged[pressure_cols].min(axis=1)
    )

    merged["pressure_max_hpa"] = (
        merged[pressure_cols].max(axis=1)
    )

    merged["pressure_range_hpa"] = (
        merged["pressure_max_hpa"]
        - merged["pressure_min_hpa"]
    )

    merged["wind_mean_ms"] = (
        merged[wind_cols].mean(axis=1)
    )

    merged["wind_min_ms"] = (
        merged[wind_cols].min(axis=1)
    )

    merged["wind_max_ms"] = (
        merged[wind_cols].max(axis=1)
    )

    merged["wind_range_ms"] = (
        merged["wind_max_ms"]
        - merged["wind_min_ms"]
    )

    # Lead time should agree across all contributing model families.
    def common_lead(row):
        values = {
            int(v)
            for v in row[lead_cols]
            if pd.notna(v)
        }

        if len(values) != 1:
            raise ValueError(
                "Model lead times disagree at "
                f"{row['valid_time']}: {sorted(values)}"
            )

        return values.pop()

    merged["lead_time_hours"] = merged.apply(
        common_lead,
        axis=1,
    )

    result = pd.DataFrame(
        {
            "storm": storm,
            "tracker": tracker,
            "valid_time": merged["valid_time"],
            "lead_time_hours":
                merged["lead_time_hours"],
            "n_models":
                merged["n_models"].astype(int),
            "availability_status":
                merged["availability_status"],
            "intensity_supported":
                merged["intensity_supported"],
            "aifs2_pressure_hpa":
                merged["aifs2_pressure_hpa"],
            "graphcast_pressure_hpa":
                merged["graphcast_pressure_hpa"],
            "pangu_pressure_hpa":
                merged["pangu_pressure_hpa"],
            "pressure_mean_hpa":
                merged["pressure_mean_hpa"],
            "pressure_range_hpa":
                merged["pressure_range_hpa"],
            "aifs2_wind_ms":
                merged["aifs2_wind_ms"],
            "graphcast_wind_ms":
                merged["graphcast_wind_ms"],
            "pangu_wind_ms":
                merged["pangu_wind_ms"],
            "wind_mean_ms":
                merged["wind_mean_ms"],
            "wind_range_ms":
                merged["wind_range_ms"],
        }
    )

    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build operational TC intensity guidance from "
            "AIFS2, GraphCast, and Pangu-Weather WuDuan tracks."
        )
    )

    parser.add_argument(
        "--aifs2",
        required=True,
        type=Path,
    )

    parser.add_argument(
        "--graphcast",
        required=True,
        type=Path,
    )

    parser.add_argument(
        "--pangu",
        required=True,
        type=Path,
    )

    parser.add_argument(
        "--storm",
        required=True,
    )

    parser.add_argument(
        "--tracker",
        default="wuduan",
    )

    parser.add_argument(
        "--output",
        required=True,
        type=Path,
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    guidance = build_intensity_guidance(
        aifs2_path=args.aifs2,
        graphcast_path=args.graphcast,
        pangu_path=args.pangu,
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
        guidance["intensity_supported"].sum()
    )

    print("Operational TC intensity guidance")
    print("---------------------------------")
    print(f"Storm             : {args.storm}")
    print(f"Tracker           : {args.tracker}")
    print(f"Guidance rows     : {len(guidance)}")
    print(
        "Supported rows    : "
        f"{supported}/{len(guidance)}"
    )
    print(f"Output            : {args.output}")

    print()
    print(
        "NOTE: intermodel intensity spread is descriptive "
        "and is not calibrated forecast confidence."
    )

    print(
        "NOTE: no retrospective intensity bias correction "
        "has been applied."
    )


if __name__ == "__main__":
    main()
