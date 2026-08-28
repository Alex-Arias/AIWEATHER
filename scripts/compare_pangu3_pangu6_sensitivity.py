#!/usr/bin/env python3
"""
Pangu3-Pangu6 cadence sensitivity analysis.

Purpose
-------
Compare Pangu3 and Pangu6 forecasts at their exact common
6-hour valid times and quantify the effect of the additional
3-hour Pangu3 forecast states on tropical-cyclone verification.

Important interpretation
------------------------
Earth2Studio Pangu3 and Pangu6 share the same 6-hour and
24-hour forecast branches at common valid times. Pangu3 adds
intermediate 3-hour states.

Therefore this script is a cadence/sampling sensitivity
analysis, not an independent-model skill comparison.

Inputs
------
Existing forecast stores:

    outputs/pangu3/<init>/forecast.zarr
    outputs/pangu6/<init>/forecast.zarr

Existing native TC verification products:

    results/verification/
        pangu3_<case>_<init>/verification_native.csv

    results/verification/
        pangu6_<case>_<init>/verification_native.csv

Outputs
-------
1. field_comparison.csv
2. tc_common_time_comparison.csv
3. sensitivity_summary.csv

The script does not rerun forecasts or TC tracking.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr


# ============================================================
# Helpers
# ============================================================

def normalize_init(value: str) -> str:
    """
    Normalize initialization time to YYYYMMDDTHHMMSS.
    """

    return (
        value
        .replace("-", "")
        .replace(":", "")
    )


def lead_hours(values) -> np.ndarray:
    """
    Convert numpy timedelta64 lead times to integer hours.
    """

    return np.asarray(
        values / np.timedelta64(1, "h"),
        dtype=int,
    )


def circular_longitude_difference(
    pangu6,
    pangu3,
):
    """
    Signed Pangu6-minus-Pangu3 longitude difference
    constrained to [-180, 180) degrees.
    """

    return (
        (
            np.asarray(
                pangu6,
                dtype=float,
            )
            - np.asarray(
                pangu3,
                dtype=float,
            )
            + 180.0
        )
        % 360.0
        - 180.0
    )


def numeric_difference_metrics(
    a: np.ndarray,
    b: np.ndarray,
):
    """
    Difference metrics for Pangu6 minus Pangu3.
    """

    a = np.asarray(a)
    b = np.asarray(b)

    if a.shape != b.shape:
        raise ValueError(
            "Cannot compare arrays with different shapes: "
            f"{a.shape} versus {b.shape}"
        )

    bitwise_identical = np.array_equal(
        a,
        b,
        equal_nan=True,
    )

    difference = (
        b.astype(
            np.float64,
            copy=False,
        )
        - a.astype(
            np.float64,
            copy=False,
        )
    )

    finite = np.isfinite(
        difference
    )

    if not finite.any():
        return {
            "mae": np.nan,
            "rmse": np.nan,
            "maximum_absolute_difference": np.nan,
            "bitwise_identical": bitwise_identical,
        }

    values = difference[
        finite
    ]

    abs_values = np.abs(
        values
    )

    return {
        "mae": float(
            np.mean(
                abs_values
            )
        ),
        "rmse": float(
            np.sqrt(
                np.mean(
                    values ** 2
                )
            )
        ),
        "maximum_absolute_difference": float(
            np.max(
                abs_values
            )
        ),
        "bitwise_identical": bool(
            bitwise_identical
        ),
    }


def default_forecast_path(
    model: str,
    init: str,
) -> Path:
    """
    Canonical AIWeather forecast path.
    """

    return (
        Path("outputs")
        / model
        / init
        / "forecast.zarr"
    )


def default_verification_path(
    model: str,
    case: str,
    init: str,
) -> Path:
    """
    Current model-specific TC verification path.
    """

    return (
        Path("results")
        / "verification"
        / f"{model}_{case}_{init}"
        / "verification_native.csv"
    )


# ============================================================
# Arguments
# ============================================================

def parse_args():

    parser = argparse.ArgumentParser(
        description=(
            "Compare Pangu3 and Pangu6 forecast states "
            "at exact common lead times and quantify "
            "TC verification cadence sensitivity."
        )
    )

    parser.add_argument(
        "--case",
        required=True,
        help=(
            "Case name used in verification paths, "
            "for example genevieve."
        ),
    )

    parser.add_argument(
        "--init",
        required=True,
        help=(
            "Initialization time as YYYYMMDDTHHMMSS "
            "or YYYY-MM-DDTHH:MM:SS."
        ),
    )

    parser.add_argument(
        "--pangu3-forecast",
        default=None,
        help="Optional explicit Pangu3 forecast.zarr path.",
    )

    parser.add_argument(
        "--pangu6-forecast",
        default=None,
        help="Optional explicit Pangu6 forecast.zarr path.",
    )

    parser.add_argument(
        "--pangu3-verification",
        default=None,
        help=(
            "Optional explicit Pangu3 "
            "verification_native.csv path."
        ),
    )

    parser.add_argument(
        "--pangu6-verification",
        default=None,
        help=(
            "Optional explicit Pangu6 "
            "verification_native.csv path."
        ),
    )

    parser.add_argument(
        "--output-dir",
        default=None,
        help="Optional explicit output directory.",
    )

    return parser.parse_args()


# ============================================================
# Main
# ============================================================

def main():

    args = parse_args()

    case = (
        args.case
        .strip()
        .lower()
    )

    init = normalize_init(
        args.init
    )

    pangu3_forecast = (
        Path(
            args.pangu3_forecast
        )
        if args.pangu3_forecast
        else default_forecast_path(
            "pangu3",
            init,
        )
    )

    pangu6_forecast = (
        Path(
            args.pangu6_forecast
        )
        if args.pangu6_forecast
        else default_forecast_path(
            "pangu6",
            init,
        )
    )

    pangu3_verification = (
        Path(
            args.pangu3_verification
        )
        if args.pangu3_verification
        else default_verification_path(
            "pangu3",
            case,
            init,
        )
    )

    pangu6_verification = (
        Path(
            args.pangu6_verification
        )
        if args.pangu6_verification
        else default_verification_path(
            "pangu6",
            case,
            init,
        )
    )

    output_dir = (
        Path(
            args.output_dir
        )
        if args.output_dir
        else (
            Path("results")
            / "sensitivity"
            / "pangu3_pangu6"
            / f"{case}_{init}"
        )
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Validate inputs
    # --------------------------------------------------------

    required = [
        pangu3_forecast,
        pangu6_forecast,
        pangu3_verification,
        pangu6_verification,
    ]

    missing = [
        path
        for path in required
        if not path.exists()
    ]

    if missing:
        message = "\n".join(
            str(path)
            for path in missing
        )

        raise FileNotFoundError(
            "Required sensitivity-analysis inputs "
            f"were not found:\n{message}"
        )

    print("=" * 78)
    print(
        "PANGU3-PANGU6 CADENCE SENSITIVITY ANALYSIS"
    )
    print("=" * 78)

    print()
    print("Case                :", case)
    print("Initialization      :", init)
    print("Pangu3 forecast     :", pangu3_forecast)
    print("Pangu6 forecast     :", pangu6_forecast)
    print("Pangu3 verification :", pangu3_verification)
    print("Pangu6 verification :", pangu6_verification)
    print("Output directory    :", output_dir)

    # ========================================================
    # Forecast-field comparison
    # ========================================================

    print()
    print("=" * 78)
    print(
        "FORECAST FIELD COMPARISON"
    )
    print("=" * 78)

    p3 = xr.open_zarr(
        pangu3_forecast
    )

    p6 = xr.open_zarr(
        pangu6_forecast
    )

    common_leads = np.intersect1d(
        p3["lead_time"].values,
        p6["lead_time"].values,
    )

    if len(common_leads) == 0:
        raise ValueError(
            "Pangu3 and Pangu6 have no common lead times."
        )

    common_vars = sorted(
        set(
            p3.data_vars
        )
        & set(
            p6.data_vars
        )
    )

    if not common_vars:
        raise ValueError(
            "Pangu3 and Pangu6 have no common variables."
        )

    print(
        "Common lead times:",
        len(
            common_leads
        ),
    )

    print(
        "Common variables :",
        len(
            common_vars
        ),
    )

    field_rows = []

    for index, variable in enumerate(
        common_vars,
        start=1,
    ):

        a = (
            p3[variable]
            .sel(
                lead_time=common_leads
            )
            .values
        )

        b = (
            p6[variable]
            .sel(
                lead_time=common_leads
            )
            .values
        )

        metrics = (
            numeric_difference_metrics(
                a,
                b,
            )
        )

        field_rows.append(
            {
                "variable":
                    variable,
                "common_lead_count":
                    len(
                        common_leads
                    ),
                "mae_pangu6_minus_pangu3":
                    metrics[
                        "mae"
                    ],
                "rmse_pangu6_minus_pangu3":
                    metrics[
                        "rmse"
                    ],
                "maximum_absolute_difference":
                    metrics[
                        "maximum_absolute_difference"
                    ],
                "bitwise_identical":
                    metrics[
                        "bitwise_identical"
                    ],
            }
        )

        status = (
            "IDENTICAL"
            if metrics[
                "bitwise_identical"
            ]
            else "DIFFERENT"
        )

        print(
            f"[{index:02d}/{len(common_vars):02d}] "
            f"{variable:10s} {status}"
        )

    field_comparison = pd.DataFrame(
        field_rows
    )

    field_output = (
        output_dir
        / "field_comparison.csv"
    )

    field_comparison.to_csv(
        field_output,
        index=False,
    )

    identical_variables = int(
        field_comparison[
            "bitwise_identical"
        ].sum()
    )

    all_fields_identical = bool(
        field_comparison[
            "bitwise_identical"
        ].all()
    )

    # ========================================================
    # TC common-time comparison
    # ========================================================

    print()
    print("=" * 78)
    print(
        "TC COMMON-TIME COMPARISON"
    )
    print("=" * 78)

    tc3 = pd.read_csv(
        pangu3_verification,
        parse_dates=[
            "valid_time",
        ],
    )

    tc6 = pd.read_csv(
        pangu6_verification,
        parse_dates=[
            "valid_time",
        ],
    )

    tc = tc3.merge(
        tc6,
        on="valid_time",
        suffixes=(
            "_pangu3",
            "_pangu6",
        ),
        how="inner",
        validate="one_to_one",
    )

    if len(tc) == 0:
        raise ValueError(
            "No exact common TC verification times "
            "were found."
        )

    tc_output = pd.DataFrame()

    tc_output[
        "valid_time"
    ] = tc[
        "valid_time"
    ]

    tc_output[
        "lead_time_hours"
    ] = tc[
        "lead_time_hours_pangu6"
    ]

    tc_output[
        "lead_time_hours_pangu3"
    ] = tc[
        "lead_time_hours_pangu3"
    ]

    tc_output[
        "lead_time_hours_pangu6"
    ] = tc[
        "lead_time_hours_pangu6"
    ]

    tc_output[
        "pangu3_latitude"
    ] = tc[
        "forecast_latitude_pangu3"
    ]

    tc_output[
        "pangu6_latitude"
    ] = tc[
        "forecast_latitude_pangu6"
    ]

    tc_output[
        "latitude_difference_deg"
    ] = (
        tc_output[
            "pangu6_latitude"
        ]
        - tc_output[
            "pangu3_latitude"
        ]
    )

    tc_output[
        "pangu3_longitude"
    ] = tc[
        "forecast_longitude_pangu3"
    ]

    tc_output[
        "pangu6_longitude"
    ] = tc[
        "forecast_longitude_pangu6"
    ]

    tc_output[
        "longitude_difference_deg"
    ] = (
        circular_longitude_difference(
            tc_output[
                "pangu6_longitude"
            ],
            tc_output[
                "pangu3_longitude"
            ],
        )
    )

    tc_output[
        "pangu3_pressure_pa"
    ] = tc[
        "forecast_pressure_pa_pangu3"
    ]

    tc_output[
        "pangu6_pressure_pa"
    ] = tc[
        "forecast_pressure_pa_pangu6"
    ]

    tc_output[
        "pressure_difference_pa"
    ] = (
        tc_output[
            "pangu6_pressure_pa"
        ]
        - tc_output[
            "pangu3_pressure_pa"
        ]
    )

    tc_output[
        "pangu3_wind_ms"
    ] = tc[
        "forecast_wind_ms_pangu3"
    ]

    tc_output[
        "pangu6_wind_ms"
    ] = tc[
        "forecast_wind_ms_pangu6"
    ]

    tc_output[
        "wind_difference_ms"
    ] = (
        tc_output[
            "pangu6_wind_ms"
        ]
        - tc_output[
            "pangu3_wind_ms"
        ]
    )

    tc_output[
        "pangu3_track_error_km"
    ] = tc[
        "track_error_km_pangu3"
    ]

    tc_output[
        "pangu6_track_error_km"
    ] = tc[
        "track_error_km_pangu6"
    ]

    tc_output[
        "track_error_difference_km"
    ] = (
        tc_output[
            "pangu6_track_error_km"
        ]
        - tc_output[
            "pangu3_track_error_km"
        ]
    )

    tc_common_output = (
        output_dir
        / "tc_common_time_comparison.csv"
    )

    tc_output.to_csv(
        tc_common_output,
        index=False,
    )

    tc_track_identical = bool(
        np.array_equal(
            tc_output[
                "pangu3_latitude"
            ].to_numpy(),
            tc_output[
                "pangu6_latitude"
            ].to_numpy(),
            equal_nan=True,
        )
        and np.array_equal(
            tc_output[
                "pangu3_longitude"
            ].to_numpy(),
            tc_output[
                "pangu6_longitude"
            ].to_numpy(),
            equal_nan=True,
        )
    )

    tc_intensity_identical = bool(
        np.array_equal(
            tc_output[
                "pangu3_pressure_pa"
            ].to_numpy(),
            tc_output[
                "pangu6_pressure_pa"
            ].to_numpy(),
            equal_nan=True,
        )
        and np.array_equal(
            tc_output[
                "pangu3_wind_ms"
            ].to_numpy(),
            tc_output[
                "pangu6_wind_ms"
            ].to_numpy(),
            equal_nan=True,
        )
    )

    tc_all_identical = bool(
        tc_track_identical
        and tc_intensity_identical
        and np.array_equal(
            tc_output[
                "pangu3_track_error_km"
            ].to_numpy(),
            tc_output[
                "pangu6_track_error_km"
            ].to_numpy(),
            equal_nan=True,
        )
    )

    # ========================================================
    # Summary
    # ========================================================

    print()
    print("=" * 78)
    print(
        "SENSITIVITY SUMMARY"
    )
    print("=" * 78)

    pangu3_total_leads = int(
        p3.sizes[
            "lead_time"
        ]
    )

    pangu6_total_leads = int(
        p6.sizes[
            "lead_time"
        ]
    )

    pangu3_lead_hours = lead_hours(
        p3[
            "lead_time"
        ].values
    )

    pangu6_lead_hours = lead_hours(
        p6[
            "lead_time"
        ].values
    )

    pangu3_native_timestep = (
        int(
            np.min(
                np.diff(
                    pangu3_lead_hours
                )
            )
        )
        if len(
            pangu3_lead_hours
        ) > 1
        else np.nan
    )

    pangu6_native_timestep = (
        int(
            np.min(
                np.diff(
                    pangu6_lead_hours
                )
            )
        )
        if len(
            pangu6_lead_hours
        ) > 1
        else np.nan
    )

    summary = pd.DataFrame(
        [
            {
                "case":
                    case,
                "initialization":
                    init,
                "pangu3_native_timestep_h":
                    pangu3_native_timestep,
                "pangu6_native_timestep_h":
                    pangu6_native_timestep,
                "pangu3_total_lead_count":
                    pangu3_total_leads,
                "pangu6_total_lead_count":
                    pangu6_total_leads,
                "common_lead_count":
                    len(
                        common_leads
                    ),
                "common_variable_count":
                    len(
                        common_vars
                    ),
                "identical_variable_count":
                    identical_variables,
                "all_fields_identical":
                    all_fields_identical,
                "pangu3_tc_point_count":
                    len(
                        tc3
                    ),
                "pangu6_tc_point_count":
                    len(
                        tc6
                    ),
                "tc_common_point_count":
                    len(
                        tc_output
                    ),
                "tc_track_identical":
                    tc_track_identical,
                "tc_intensity_identical":
                    tc_intensity_identical,
                "tc_all_common_values_identical":
                    tc_all_identical,
                "maximum_common_track_difference_km":
                    float(
                        np.abs(
                            tc_output[
                                "track_error_difference_km"
                            ]
                        ).max()
                    ),
                "maximum_common_pressure_difference_pa":
                    float(
                        np.abs(
                            tc_output[
                                "pressure_difference_pa"
                            ]
                        ).max()
                    ),
                "maximum_common_wind_difference_ms":
                    float(
                        np.abs(
                            tc_output[
                                "wind_difference_ms"
                            ]
                        ).max()
                    ),
                "interpretation":
                    (
                        "cadence sensitivity; Pangu3 adds "
                        "intermediate 3-hour states to the "
                        "shared 6-hour/24-hour trajectory"
                    ),
            }
        ]
    )

    summary_output = (
        output_dir
        / "sensitivity_summary.csv"
    )

    summary.to_csv(
        summary_output,
        index=False,
    )

    print()
    print(
        "All forecast fields identical:",
        all_fields_identical,
    )

    print(
        "Identical variables:",
        f"{identical_variables}/{len(common_vars)}",
    )

    print(
        "Pangu3 forecast states:",
        pangu3_total_leads,
    )

    print(
        "Pangu6 forecast states:",
        pangu6_total_leads,
    )

    print(
        "Common forecast states:",
        len(
            common_leads
        ),
    )

    print(
        "Pangu3 TC points:",
        len(
            tc3
        ),
    )

    print(
        "Pangu6 TC points:",
        len(
            tc6
        ),
    )

    print(
        "Common TC points:",
        len(
            tc_output
        ),
    )

    print(
        "TC track identical:",
        tc_track_identical,
    )

    print(
        "TC intensity identical:",
        tc_intensity_identical,
    )

    print(
        "All common TC values identical:",
        tc_all_identical,
    )

    print()
    print("Products:")
    print(" ", field_output)
    print(" ", tc_common_output)
    print(" ", summary_output)


if __name__ == "__main__":
    main()
