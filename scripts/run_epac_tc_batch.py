#!/usr/bin/env python3
"""
Run the five-storm eastern Pacific tropical-cyclone
verification batch for GraphCast, AIFS2, or Pangu3.

Storms
------
Douglas
Elida
Fausto
Genevieve
Hernan
"""


import argparse
from pathlib import Path

from aiweather.verification import (
    TCVerificationCase,
    run_tc_verification_batch,
)


# ============================================================
# Configuration
# ============================================================

MINIMUM_OVERLAP = 6
MAXIMUM_MEAN_ERROR_KM = 450.0

FIELD_LEAD_TIMES = [
    24,
    48,
    72,
    96,
]


STORMS = [

    {
        "case_id": "douglas_20260701T000000",
        "sid": "2026180N11238",
        "init": "20260701T000000",
        "lat_min": 5.0,
        "lat_max": 30.0,
        "lon_min": -145.0,
        "lon_max": -110.0,
    },

    {
        "case_id": "elida_20260714T120000",
        "sid": "2026194N12265",
        "init": "20260714T120000",
        "lat_min": 5.0,
        "lat_max": 35.0,
        "lon_min": -140.0,
        "lon_max": -90.0,
    },
    {
        "case_id": "fausto_20260719T000000",
        "sid": "2026198N08260",
        "init": "20260719T000000",
        "lat_min": 5.0,
        "lat_max": 35.0,
        "lon_min": -150.0,
        "lon_max": -95.0,
    },
    {
        "case_id": "genevieve_20260724T000000",
        "sid": "2026204N08267",
        "init": "20260724T000000",
        "lat_min": 5.0,
        "lat_max": 35.0,
        "lon_min": -130.0,
        "lon_max": -90.0,
    },
    {
        "case_id": "hernan_20260811T000000",
        "sid": "2026223N14233",
        "init": "20260811T000000",
        "lat_min": 5.0,
        "lat_max": 35.0,
        "lon_min": -150.0,
        "lon_max": -100.0,
    },
]


def build_cases(
    model: str,
) -> list[TCVerificationCase]:
    """
    Construct the standard five-storm EPAC batch.
    """

    cases = []

    for storm in STORMS:
        cases.append(
            TCVerificationCase(
                forecast_path=(
                    Path("outputs")
                    / model
                    / storm["init"]
                    / "forecast.zarr"
                ),
                sid=storm["sid"],
                lat_min=storm["lat_min"],
                lat_max=storm["lat_max"],
                lon_min=storm["lon_min"],
                lon_max=storm["lon_max"],
                case_id=storm["case_id"],
            )
        )

    return cases


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run the five-storm EPAC tropical-cyclone "
            "verification batch."
        )
    )

    parser.add_argument(
        "model",
        choices=[
            "aifs2",
            "graphcast",
            "pangu3",
        ],
        help="Forecast model to verify.",
    )

    parser.add_argument(
        "--device",
        default="cpu",
        help=(
            "Device used by Earth2Studio trackers. "
            "Default: cpu."
        ),
    )

    parser.add_argument(
        "--plots",
        action="store_true",
        help=(
            "Generate per-case verification plots."
        ),
    )

    args = parser.parse_args()

    model = args.model

    cases = build_cases(
        model
    )

    output_dir = (
        Path("results")
        / "verification"
        / "batch"
        / f"{model}_epac_5storm"
    )

    print("=" * 76)
    print(
        f"{model.upper()} — EPAC FIVE-STORM "
        "TC VERIFICATION"
    )
    print("=" * 76)

    print()
    print(
        "Minimum overlap       :",
        MINIMUM_OVERLAP,
    )
    print(
        "Maximum mean error km :",
        MAXIMUM_MEAN_ERROR_KM,
    )
    print(
        "Output directory      :",
        output_dir,
    )

    print()
    print("Cases:")

    for case in cases:
        print(
            f"  {case.case_id:<28s} "
            f"{case.sid}"
        )
        print(
            f"      {case.forecast_path}"
        )

    print()

    result = run_tc_verification_batch(
        cases,
        device=args.device,
        minimum_overlap=(
            MINIMUM_OVERLAP
        ),
        maximum_mean_error_km=(
            MAXIMUM_MEAN_ERROR_KM
        ),
        generate_plots=args.plots,
        field_lead_times=(
            FIELD_LEAD_TIMES
        ),
        ibtracs_basin="EP",
        output_dir=output_dir,
    )

    print()
    print("=" * 76)
    print("BATCH VERIFICATION COMPLETE")
    print("=" * 76)

    print()
    print("Summary:")
    print(result.summary_path)

    print("Aggregate:")
    print(result.aggregate_path)

    print("Lead time:")
    print(result.lead_time_path)

    print("Points:")
    print(result.points_path)

    print()
    print(
        result.summary.to_string(
            index=False,
        )
    )


if __name__ == "__main__":
    main()
