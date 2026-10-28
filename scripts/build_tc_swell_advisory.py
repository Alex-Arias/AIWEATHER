#!/usr/bin/env python3
"""Integrate existing AIWeather TC guidance products.

Stage 1: combine position and intensity guidance without
altering the underlying scientific estimates.

Coastal-wave integration and TC-swell attribution are
intentionally excluded from this initial implementation.
"""

import argparse
import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.build_operational_tc_guidance import (
    availability_status,
    parse_bool_series,
)
from scripts.tc_swell_spectral_events import build_spectral_event_guidance


KEYS = ["storm", "tracker", "valid_time", "lead_time_hours"]


def read_guidance(path: Path, required: set[str]) -> pd.DataFrame:
    """Read guidance and validate its identifying columns."""

    if not path.is_file():
        raise FileNotFoundError(path)

    df = pd.read_csv(path)

    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"{path}: missing columns {sorted(missing)}"
        )

    df = df.copy()

    df["valid_time"] = pd.to_datetime(
        df["valid_time"], errors="raise"
    )

    df["lead_time_hours"] = pd.to_numeric(
        df["lead_time_hours"], errors="raise"
    )

    if df[KEYS].isna().any().any():
        raise ValueError(f"{path}: missing identifying values")

    if df.duplicated(KEYS).any():
        raise ValueError(f"{path}: duplicate guidance keys")

    return df


def validate_forecast_times(
    df: pd.DataFrame,
    init: str,
    name: str,
) -> None:
    """Validate forecast timestamps against initialization."""

    if df.empty:
        raise ValueError(f"{name}: empty guidance dataset")

    try:
        initialization = pd.to_datetime(
            init,
            format="%Y%m%dT%H%M%S",
            errors="raise",
        )
    except (ValueError, TypeError) as exc:
        raise ValueError(
            f"Invalid forecast initialization: {init!r}"
        ) from exc

    lead = pd.to_numeric(
        df["lead_time_hours"],
        errors="raise",
    )

    if not np.isfinite(lead.to_numpy(dtype=float)).all():
        raise ValueError(
            f"{name}: nonfinite forecast lead times"
        )

    if (lead < 0).any():
        raise ValueError(
            f"{name}: negative forecast lead times"
        )

    valid = pd.to_datetime(
        df["valid_time"],
        errors="raise",
    )

    if valid.isna().any():
        raise ValueError(
            f"{name}: missing forecast timestamps"
        )

    if valid.duplicated().any():
        raise ValueError(
            f"{name}: duplicate forecast timestamps"
        )

    if lead.duplicated().any():
        raise ValueError(
            f"{name}: duplicate forecast lead times"
        )

    expected = (
        initialization
        + pd.to_timedelta(lead, unit="h")
    )

    if not valid.eq(expected).all():
        raise ValueError(
            f"{name}: valid_time does not match "
            "initialization + lead_time_hours"
        )


def build_tc_guidance(
    track: pd.DataFrame,
    intensity: pd.DataFrame,
) -> pd.DataFrame:
    """Combine TC position and intensity by exact identifiers."""

    merged = track.merge(
        intensity,
        on=KEYS,
        how="outer",
        suffixes=("_track", "_intensity"),
        indicator=True,
        validate="one_to_one",
    )

    if not merged["_merge"].eq("both").all():
        raise ValueError(
            "Track and intensity guidance timestamps differ"
        )

    merged = merged.drop(columns="_merge")

    if not (
        merged["n_model_families"]
        .eq(merged["n_models"])
        .all()
    ):
        raise ValueError(
            "Position and intensity model-family counts differ"
        )

    for column in [
        "guidance_supported",
        "intensity_supported",
    ]:
        if merged[column].isna().any():
            raise ValueError(
                f"Missing support indicator: {column}"
            )

    for column in [
        "guidance_supported",
        "intensity_supported",
    ]:
        try:
            merged[column] = parse_bool_series(
                merged[column]
            )
        except ValueError as exc:
            raise ValueError(
                f"Invalid support indicator in {column}: {exc}"
            ) from exc

    counts = pd.to_numeric(
        merged["n_model_families"],
        errors="coerce",
    )

    if (
        counts.isna().any()
        or not counts.isin([1, 2, 3]).all()
    ):
        raise ValueError(
            "Invalid independent model-family count; "
            "expected integer values 1, 2, or 3"
        )

    merged["n_model_families"] = counts.astype(int)

    availability = merged["n_model_families"].map(
        availability_status
    )

    merged["tc_availability_status"] = (
        availability.str.lower()
    )

    insufficient_support = (
        merged["n_model_families"] < 3
    ) & (
        merged["guidance_supported"]
        | merged["intensity_supported"]
    )

    if insufficient_support.any():
        raise ValueError(
            "Upstream scientific support is inconsistent "
            "with independent model-family availability"
        )

    merged["tc_jointly_supported"] = (
        merged["guidance_supported"]
        & merged["intensity_supported"]
        & merged["n_model_families"].eq(3)
    )

    merged["tc_support_status"] = (
        merged["tc_availability_status"]
    )

    merged.loc[
        merged["n_model_families"].eq(3)
        & ~merged["tc_jointly_supported"],
        "tc_support_status",
    ] = "insufficient"

    merged = merged.sort_values(
        ["valid_time", "lead_time_hours"]
    ).reset_index(drop=True)

    return merged


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Integrate existing AIWeather TC position "
            "and intensity guidance."
        )
    )

    parser.add_argument(
        "--mode",
        choices=("tc", "coastal-wave", "spectral-event"),
        default="tc",
        help="Advisory integration product (default: tc)",
    )

    parser.add_argument("--storm", required=True)
    parser.add_argument("--init", required=True)
    parser.add_argument("--tracker", default="wuduan")
    parser.add_argument("--output", type=Path)

    return parser.parse_args()





def validate_coastal_wave_inputs(
    wave_data: pd.DataFrame,
    tc_data: pd.DataFrame,
) -> None:
    """Validate forecast timing and TC support before integration."""

    inferred_initializations = {}

    for name, frame in (
        ("wave", wave_data),
        ("tc", tc_data),
    ):
        leads = pd.to_numeric(
            frame["lead_time_hours"],
            errors="raise",
        )

        if (
            not np.isfinite(leads.to_numpy(dtype=float)).all()
            or (leads < 0).any()
        ):
            raise ValueError(
                f"{name}: invalid or negative forecast lead"
            )

        times = pd.to_datetime(
            frame["valid_time"],
            errors="raise",
        )

        if times.isna().any():
            raise ValueError(
                f"{name}: missing forecast timestamp"
            )

        inferred = (
            times
            - pd.to_timedelta(leads, unit="h")
        )

        if inferred.isna().any() or inferred.nunique() != 1:
            raise ValueError(
                f"{name}: inconsistent forecast initialization"
            )

        inferred_initializations[name] = inferred.iloc[0]

    if (
        inferred_initializations["wave"]
        != inferred_initializations["tc"]
    ):
        raise ValueError(
            "Wave and TC forecast initializations differ"
        )

    support = tc_data["tc_jointly_supported"]

    if support.isna().any():
        raise ValueError(
            "Available TC guidance has missing joint support"
        )

    support = parse_bool_series(
        support,
    ).astype(bool)

    statuses = tc_data["tc_support_status"]

    allowed = {
        "supported",
        "limited",
        "insufficient",
    }

    if statuses.isna().any():
        raise ValueError(
            "Available TC guidance has missing support status"
        )

    if not statuses.isin(allowed).all():
        raise ValueError(
            "Invalid TC support-status classification"
        )

    inconsistent = (
        (support & statuses.ne("supported"))
        | (~support & statuses.eq("supported"))
    )

    if inconsistent.any():
        raise ValueError(
            "TC joint support and support status disagree"
        )


def build_coastal_wave_guidance(
    wave: pd.DataFrame,
    tc: pd.DataFrame,
) -> pd.DataFrame:
    """Integrate AIFS2 coastal-wave and available TC guidance.

    Preserve the full wave forecast horizon. TC fields remain
    missing when corresponding TC guidance is unavailable.

    Temporal coincidence does not establish storm attribution.
    """

    wave_keys = [
        "sector",
        "valid_time",
        "lead_time_hours",
    ]

    tc_keys = [
        "valid_time",
        "lead_time_hours",
    ]

    tc_fields = {
        "tc_jointly_supported": "tc_jointly_supported",
        "tc_support_status": "tc_support_status",
        "consensus_latitude": "tc_latitude",
        "consensus_longitude": "tc_longitude",
        "pressure_mean_hpa": "tc_pressure_hpa",
        "wind_mean_ms": "tc_wind_ms",
        "r67_km": "tc_r67_km",
        "r90_km": "tc_r90_km",
    }

    required_wave = set(wave_keys)

    required_tc = (
        set(tc_keys)
        | set(tc_fields)
    )

    missing_wave = required_wave - set(wave.columns)
    missing_tc = required_tc - set(tc.columns)

    if missing_wave:
        raise ValueError(
            f"Missing wave columns: {sorted(missing_wave)}"
        )

    if missing_tc:
        raise ValueError(
            f"Missing TC columns: {sorted(missing_tc)}"
        )

    if wave.empty or tc.empty:
        raise ValueError(
            "Wave and TC guidance must be nonempty"
        )

    if wave.duplicated(wave_keys).any():
        raise ValueError(
            "Duplicate coastal-wave integration keys"
        )

    if tc.duplicated(tc_keys).any():
        raise ValueError(
            "Duplicate TC integration keys"
        )

    wave_data = wave.copy()
    tc_data = tc[
        tc_keys + list(tc_fields)
    ].copy()

    for frame in [wave_data, tc_data]:
        frame["valid_time"] = pd.to_datetime(
            frame["valid_time"],
            errors="raise",
        )

        frame["lead_time_hours"] = pd.to_numeric(
            frame["lead_time_hours"],
            errors="raise",
        )

    if wave_data.duplicated(wave_keys).any():
        raise ValueError(
            "Duplicate normalized coastal-wave keys"
        )

    if tc_data.duplicated(tc_keys).any():
        raise ValueError(
            "Duplicate normalized TC keys"
        )

    validate_coastal_wave_inputs(
        wave_data,
        tc_data,
    )

    tc_data = tc_data.rename(
        columns=tc_fields
    )

    overlapping = (
        set(wave_data.columns)
        & set(tc_data.columns)
    ) - set(tc_keys)

    if overlapping:
        raise ValueError(
            "Unexpected wave/TC column overlap: "
            f"{sorted(overlapping)}"
        )

    result = wave_data.merge(
        tc_data,
        on=tc_keys,
        how="left",
        validate="many_to_one",
        indicator=True,
        sort=False,
    )

    if len(result) != len(wave_data):
        raise ValueError(
            "Coastal-wave record count changed"
        )

    result["tc_guidance_available"] = (
        result["_merge"] == "both"
    )

    result["tc_support_status"] = (
        result["tc_support_status"]
        .fillna("unavailable")
    )

    result["tc_jointly_supported"] = (
        result["tc_jointly_supported"]
        .astype("boolean")
    )

    result.loc[
        ~result["tc_guidance_available"],
        "tc_jointly_supported",
    ] = pd.NA

    result = result.drop(
        columns=["_merge"]
    )

    if result.duplicated(wave_keys).any():
        raise ValueError(
            "Duplicate integrated coastal-wave keys"
        )

    return result


def write_csv_exclusive(
    dataframe: pd.DataFrame,
    output: Path,
) -> None:
    """Atomically publish a completed CSV without overwriting."""

    if output.exists() or output.is_symlink():
        raise FileExistsError(
            f"STOP — Output already exists: {output}"
        )

    output.parent.mkdir(parents=True, exist_ok=True)

    temporary = None

    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="",
            suffix=".tmp",
            prefix=f".{output.name}.",
            dir=output.parent,
            delete=False,
        ) as handle:

            temporary = Path(handle.name)

            dataframe.to_csv(
                handle,
                index=False,
            )

            handle.flush()
            os.fsync(handle.fileno())

        # Hard-link publication is exclusive:
        # an existing destination is never replaced.
        os.link(
            temporary,
            output,
            follow_symlinks=False,
        )

    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)



def run_coastal_wave_mode(args):
    """Build experimental B1 coastal-wave guidance."""

    root = Path("results/operational") / (
        f"{args.storm}_{args.init}"
    ) / "advisory"

    wave_path = root / (
        f"{args.storm}_aifs2_spectral_guidance.csv"
    )

    tc_path = root / (
        f"{args.storm}_{args.tracker}_integrated_tc_guidance.csv"
    )

    output = args.output or root / (
        f"{args.storm}_{args.tracker}_coastal_wave_guidance.csv"
    )

    if output.exists() or output.is_symlink():
        raise SystemExit(
            f"STOP — Output already exists: {output}"
        )

    wave = pd.read_csv(wave_path)
    tc = pd.read_csv(tc_path)

    result = build_coastal_wave_guidance(wave, tc)

    initialization = pd.to_datetime(
        args.init,
        format="%Y%m%dT%H%M%S",
        errors="raise",
    )

    for name, frame in (
        ("wave", wave),
        ("tc", tc),
    ):
        leads = pd.to_numeric(
            frame["lead_time_hours"],
            errors="raise",
        )

        times = pd.to_datetime(
            frame["valid_time"],
            errors="raise",
        )

        expected = (
            initialization
            + pd.to_timedelta(leads, unit="h")
        )

        if not times.eq(expected).all():
            raise ValueError(
                f"{name}: timestamps do not match "
                "requested initialization"
            )

    required_provenance = {"storm", "tracker"}

    missing_provenance = (
        required_provenance - set(tc.columns)
    )

    if missing_provenance:
        raise ValueError(
            "TC guidance missing provenance columns: "
            f"{sorted(missing_provenance)}"
        )

    for column, expected_value in (
        ("storm", args.storm),
        ("tracker", args.tracker),
    ):
        if tc[column].isna().any():
            raise ValueError(
                f"TC guidance has missing {column} identifiers"
            )

        if not tc[column].eq(expected_value).all():
            raise ValueError(
                f"TC guidance {column} identifier mismatch"
            )

    write_csv_exclusive(result, output)

    print("Experimental coastal-wave guidance")
    print("----------------------------------")
    print("Storm:", args.storm)
    print("Initialization:", args.init)
    print("Tracker:", args.tracker)
    print("Wave records:", len(result))
    print(
        "TC guidance available:",
        int(result["tc_guidance_available"].sum()),
    )
    print("Output:", output)

    print(
        "\\nNOTE: Temporal coincidence does not establish "
        "storm-generated swell attribution."
    )



def run_spectral_event_mode(args):
    """Build experimental Phase B2 spectral-event guidance."""

    initialization = pd.to_datetime(
        args.init,
        format="%Y%m%dT%H%M%S",
        errors="raise",
        exact=True,
    )

    if initialization.strftime("%Y%m%dT%H%M%S") != args.init:
        raise ValueError(
            "Initialization must use YYYYMMDDTHHMMSS"
        )

    root = Path("results/operational") / (
        f"{args.storm}_{args.init}"
    ) / "advisory"

    events_path = root / (
        f"{args.storm}_aifs2_spectral_events.csv"
    )

    quality_path = root / (
        f"{args.storm}_aifs2_spectral_quality.csv"
    )

    b1_path = root / (
        f"{args.storm}_{args.tracker}_coastal_wave_guidance.csv"
    )

    output = args.output or root / (
        f"{args.storm}_{args.tracker}_spectral_event_guidance.csv"
    )

    if output.exists() or output.is_symlink():
        raise SystemExit(
            f"STOP — Output already exists: {output}"
        )

    events = pd.read_csv(events_path)
    quality = pd.read_csv(quality_path)
    b1 = pd.read_csv(b1_path)

    leads = pd.to_numeric(
        b1["lead_time_hours"],
        errors="raise",
    )

    times = pd.to_datetime(
        b1["valid_time"],
        errors="raise",
    )

    expected_times = (
        initialization
        + pd.to_timedelta(leads, unit="h")
    )

    if not times.eq(expected_times).all():
        raise ValueError(
            "B1 timestamps do not match requested initialization"
        )

    result = build_spectral_event_guidance(
        events,
        quality,
        b1,
    )

    write_csv_exclusive(result, output)

    print("Experimental spectral-event guidance")
    print("------------------------------------")
    print("Storm:", args.storm)
    print("Initialization:", args.init)
    print("Tracker:", args.tracker)
    print("Spectral events:", len(result))

    print("\nTC horizon classification:")
    print(
        result["event_tc_horizon_class"]
        .value_counts()
        .to_string()
    )

    print("\nOutput:", output)

    print(
        "\nNOTE: Temporal coincidence does not establish "
        "storm-generated swell attribution."
    )

def main():
    args = parse_args()

    if args.mode == "coastal-wave":
        return run_coastal_wave_mode(args)

    if args.mode == "spectral-event":
        return run_spectral_event_mode(args)

    root = Path("results/operational") / (
        f"{args.storm}_{args.init}"
    ) / "advisory"

    track_path = root / (
        f"{args.storm}_{args.tracker}_tc_guidance.csv"
    )

    intensity_path = root / (
        f"{args.storm}_{args.tracker}_intensity_guidance.csv"
    )

    output = args.output or root / (
        f"{args.storm}_{args.tracker}_integrated_tc_guidance.csv"
    )

    if output.exists():
        raise SystemExit(
            f"STOP — Output already exists: {output}"
        )

    track = read_guidance(
        track_path,
        set(KEYS) | {
            "n_model_families",
            "guidance_supported",
            "consensus_latitude",
            "consensus_longitude",
            "r67_km",
            "r90_km",
        },
    )

    intensity = read_guidance(
        intensity_path,
        set(KEYS) | {
            "n_models",
            "intensity_supported",
            "pressure_mean_hpa",
            "wind_mean_ms",
        },
    )

    for name, df in [
        ("track", track),
        ("intensity", intensity),
    ]:
        if not df["storm"].eq(args.storm).all():
            raise ValueError(
                f"{name}: unexpected storm identifier"
            )

        if not df["tracker"].eq(args.tracker).all():
            raise ValueError(
                f"{name}: unexpected tracker identifier"
            )

    validate_forecast_times(
        track,
        args.init,
        "track",
    )

    validate_forecast_times(
        intensity,
        args.init,
        "intensity",
    )

    result = build_tc_guidance(track, intensity)

    write_csv_exclusive(result, output)

    print("Integrated TC guidance")
    print("----------------------")
    print("Storm:", args.storm)
    print("Initialization:", args.init)
    print("Tracker:", args.tracker)
    print("Records:", len(result))

    print("\nSupport status:")
    print(
        result["tc_support_status"]
        .value_counts()
        .to_string()
    )

    print("\nOutput:", output)

    print(
        "\nNOTE: Intermodel spread is descriptive, "
        "not calibrated forecast confidence."
    )


if __name__ == "__main__":
    main()
