#!/usr/bin/env python3
"""
Pangu temporal-sampling sensitivity analysis.

Reads existing AIWeather verification and pre-genesis CSV products only.
It does not rerun forecasts, tracking, or radial diagnostics.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def normalize_init(value: str) -> str:
    return value.strip().replace("-", "").replace(":", "")


def rmse(values) -> float:
    values = np.asarray(values, dtype=float)
    finite = np.isfinite(values)
    if not finite.any():
        return np.nan
    return float(np.sqrt(np.mean(values[finite] ** 2)))


def first_true_row(frame: pd.DataFrame, column: str):
    mask = frame[column].astype(str).str.lower().eq("true")
    selected = frame.loc[mask]
    return None if selected.empty else selected.iloc[0]


def first_threshold_lead(
    frame: pd.DataFrame,
    column: str,
    threshold: float,
):
    values = pd.to_numeric(frame[column], errors="coerce")
    selected = frame.loc[values >= threshold]
    if selected.empty:
        return np.nan
    return int(selected.iloc[0]["lead_time_hours"])


def gap_metrics(leads, expected_step: int):
    leads = np.asarray(
        sorted(
            pd.to_numeric(pd.Series(leads), errors="coerce")
            .dropna()
            .astype(int)
            .unique()
        ),
        dtype=int,
    )
    if len(leads) <= 1:
        return 0, 0
    gaps = np.diff(leads)
    gaps = gaps[gaps > expected_step]
    if len(gaps) == 0:
        return 0, 0
    return int(len(gaps)), int(gaps.max())


def tracker_metrics(frame: pd.DataFrame, cadence_hours: int):
    frame = frame.sort_values("lead_time_hours").reset_index(drop=True)
    gap_count, maximum_gap = gap_metrics(
        frame["lead_time_hours"],
        cadence_hours,
    )

    if frame.empty:
        return {
            "cadence_hours": cadence_hours,
            "point_count": 0,
            "first_lead_hours": np.nan,
            "last_lead_hours": np.nan,
            "coverage_duration_hours": np.nan,
            "gap_count": 0,
            "maximum_gap_hours": 0,
            "mean_track_error_km": np.nan,
            "rmse_track_error_km": np.nan,
            "median_track_error_km": np.nan,
            "maximum_track_error_km": np.nan,
            "minimum_pressure_pa": np.nan,
            "minimum_pressure_lead_hours": np.nan,
            "maximum_wind_ms": np.nan,
            "maximum_wind_lead_hours": np.nan,
        }

    first_lead = int(frame["lead_time_hours"].min())
    last_lead = int(frame["lead_time_hours"].max())
    pressure_index = frame["forecast_pressure_pa"].idxmin()
    wind_index = frame["forecast_wind_ms"].idxmax()

    return {
        "cadence_hours": cadence_hours,
        "point_count": int(len(frame)),
        "first_lead_hours": first_lead,
        "last_lead_hours": last_lead,
        "coverage_duration_hours": last_lead - first_lead,
        "gap_count": gap_count,
        "maximum_gap_hours": maximum_gap,
        "mean_track_error_km": float(frame["track_error_km"].mean()),
        "rmse_track_error_km": rmse(frame["track_error_km"]),
        "median_track_error_km": float(frame["track_error_km"].median()),
        "maximum_track_error_km": float(frame["track_error_km"].max()),
        "minimum_pressure_pa": float(
            frame.loc[pressure_index, "forecast_pressure_pa"]
        ),
        "minimum_pressure_lead_hours": int(
            frame.loc[pressure_index, "lead_time_hours"]
        ),
        "maximum_wind_ms": float(
            frame.loc[wind_index, "forecast_wind_ms"]
        ),
        "maximum_wind_lead_hours": int(
            frame.loc[wind_index, "lead_time_hours"]
        ),
    }


def pregenesis_metrics(frame: pd.DataFrame, cadence_hours: int):
    frame = frame.sort_values("lead_time_hours").reset_index(drop=True)
    first = first_true_row(frame, "organized_vortex")

    return {
        "cadence_hours": cadence_hours,
        "diagnostic_point_count": int(len(frame)),
        "first_organized_lead_hours": (
            np.nan if first is None else int(first["lead_time_hours"])
        ),
        "first_organized_valid_time": (
            None if first is None else first["valid_time"]
        ),
        "first_zeta_threshold_lead_hours": first_threshold_lead(
            frame,
            "maximum_zeta850_1e5_s1",
            15.0,
        ),
        "first_pressure_deficit_threshold_lead_hours": (
            first_threshold_lead(
                frame,
                "pressure_deficit_hpa",
                2.0,
            )
        ),
    }


def common_pregenesis_identity(
    frame_3h: pd.DataFrame,
    frame_6h: pd.DataFrame,
):
    common = frame_3h.merge(
        frame_6h,
        on="lead_time_hours",
        suffixes=("_3h", "_6h"),
        how="inner",
        validate="one_to_one",
    )

    columns = [
        "minimum_mslp_hpa",
        "environment_mslp_hpa",
        "pressure_deficit_hpa",
        "pressure_center_error_km",
        "maximum_wind10_ms",
        "maximum_zeta850_s1",
        "maximum_zeta850_1e5_s1",
        "mean_positive_zeta850_s1",
        "mean_positive_zeta850_1e5_s1",
        "zeta_center_error_km",
        "pressure_zeta_separation_km",
    ]

    rows = []
    all_identical = True

    for column in columns:
        left = common[f"{column}_3h"].to_numpy()
        right = common[f"{column}_6h"].to_numpy()
        identical = np.array_equal(left, right, equal_nan=True)
        all_identical = all_identical and identical

        difference = right.astype(float) - left.astype(float)
        finite = np.isfinite(difference)

        rows.append(
            {
                "variable": column,
                "common_point_count": len(common),
                "bitwise_identical": bool(identical),
                "maximum_absolute_difference": (
                    float(np.max(np.abs(difference[finite])))
                    if finite.any()
                    else np.nan
                ),
            }
        )

    organized_identical = np.array_equal(
        common["organized_vortex_3h"].astype(str).to_numpy(),
        common["organized_vortex_6h"].astype(str).to_numpy(),
    )
    rows.append(
        {
            "variable": "organized_vortex",
            "common_point_count": len(common),
            "bitwise_identical": bool(organized_identical),
            "maximum_absolute_difference": np.nan,
        }
    )

    all_identical = all_identical and organized_identical

    return pd.DataFrame(rows), bool(all_identical), int(len(common))


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Analyze Pangu3 3-hour versus 6-hour temporal-sampling "
            "sensitivity using existing AIWeather products."
        )
    )
    parser.add_argument("--case", required=True)
    parser.add_argument("--init", required=True)
    parser.add_argument(
        "--tracker",
        default="wuduan",
        choices=["native", "wuduan"],
    )
    parser.add_argument("--verification", default=None)
    parser.add_argument("--native-verification", default=None)
    parser.add_argument("--pregenesis-3h", default=None)
    parser.add_argument("--pregenesis-6h", default=None)
    parser.add_argument("--output-dir", default=None)
    return parser.parse_args()


def main():
    args = parse_args()

    case = args.case.strip().lower()
    init = normalize_init(args.init)
    case_id = f"{case}_{init}"

    verification = (
        Path(args.verification)
        if args.verification
        else (
            Path("results")
            / "verification"
            / f"pangu3_{case_id}"
            / f"verification_{args.tracker}.csv"
        )
    )

    native_verification = (
        Path(args.native_verification)
        if args.native_verification
        else (
            Path("results")
            / "verification"
            / f"pangu3_{case_id}"
            / "verification_native.csv"
        )
    )

    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else (
            Path("results")
            / "sensitivity"
            / "pangu_temporal_sampling"
            / case_id
        )
    )

    pregenesis_3h = (
        Path(args.pregenesis_3h)
        if args.pregenesis_3h
        else output_dir / "pregenesis_3h.csv"
    )
    pregenesis_6h = (
        Path(args.pregenesis_6h)
        if args.pregenesis_6h
        else output_dir / "pregenesis_6h.csv"
    )

    required = [verification, pregenesis_3h, pregenesis_6h]
    missing = [path for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Required existing products were not found:\n"
            + "\n".join(str(path) for path in missing)
        )

    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 78)
    print("PANGU TEMPORAL-SAMPLING SENSITIVITY")
    print("=" * 78)
    print("Case                 :", case)
    print("Initialization       :", init)
    print("Tracker              :", args.tracker)
    print("Verification         :", verification)
    print("Pre-genesis 3 h      :", pregenesis_3h)
    print("Pre-genesis 6 h      :", pregenesis_6h)
    print("Output directory     :", output_dir)

    tracker_full = pd.read_csv(verification)
    tracker_6h = tracker_full.loc[
        tracker_full["lead_time_hours"] % 6 == 0
    ].copy()

    tracker_3h_metrics = tracker_metrics(tracker_full, 3)
    tracker_6h_metrics = tracker_metrics(tracker_6h, 6)

    tracker_comparison = pd.DataFrame(
        [
            {"sampling": "3h", **tracker_3h_metrics},
            {"sampling": "6h", **tracker_6h_metrics},
        ]
    )
    tracker_output = output_dir / "tracker_sampling_comparison.csv"
    tracker_comparison.to_csv(tracker_output, index=False)

    pre3 = pd.read_csv(pregenesis_3h)
    pre6 = pd.read_csv(pregenesis_6h)

    pre3_metrics = pregenesis_metrics(pre3, 3)
    pre6_metrics = pregenesis_metrics(pre6, 6)

    pregenesis_comparison = pd.DataFrame(
        [
            {"sampling": "3h", **pre3_metrics},
            {"sampling": "6h", **pre6_metrics},
        ]
    )
    pregenesis_output = (
        output_dir / "pregenesis_sampling_comparison.csv"
    )
    pregenesis_comparison.to_csv(pregenesis_output, index=False)

    (
        pregenesis_identity,
        all_common_pregenesis_identical,
        common_pregenesis_point_count,
    ) = common_pregenesis_identity(pre3, pre6)

    identity_output = output_dir / "pregenesis_common_time_identity.csv"
    pregenesis_identity.to_csv(identity_output, index=False)

    if native_verification.exists():
        native_point_count = int(len(pd.read_csv(native_verification)))
    else:
        native_point_count = np.nan

    point_reduction = (
        tracker_3h_metrics["point_count"]
        - tracker_6h_metrics["point_count"]
    )
    point_reduction_percent = (
        100.0 * point_reduction / tracker_3h_metrics["point_count"]
        if tracker_3h_metrics["point_count"]
        else np.nan
    )

    mean_error_change = (
        tracker_6h_metrics["mean_track_error_km"]
        - tracker_3h_metrics["mean_track_error_km"]
    )
    rmse_error_change = (
        tracker_6h_metrics["rmse_track_error_km"]
        - tracker_3h_metrics["rmse_track_error_km"]
    )

    first_organized_3h = pre3_metrics["first_organized_lead_hours"]
    first_organized_6h = pre6_metrics["first_organized_lead_hours"]
    timing_change = (
        first_organized_6h - first_organized_3h
        if np.isfinite(first_organized_3h)
        and np.isfinite(first_organized_6h)
        else np.nan
    )

    intensity_extrema_identical = bool(
        np.isclose(
            tracker_3h_metrics["minimum_pressure_pa"],
            tracker_6h_metrics["minimum_pressure_pa"],
            equal_nan=True,
        )
        and (
            tracker_3h_metrics["minimum_pressure_lead_hours"]
            == tracker_6h_metrics["minimum_pressure_lead_hours"]
        )
        and np.isclose(
            tracker_3h_metrics["maximum_wind_ms"],
            tracker_6h_metrics["maximum_wind_ms"],
            equal_nan=True,
        )
        and (
            tracker_3h_metrics["maximum_wind_lead_hours"]
            == tracker_6h_metrics["maximum_wind_lead_hours"]
        )
    )

    summary = pd.DataFrame(
        [
            {
                "case": case,
                "initialization": init,
                "tracker": args.tracker,
                "native_tracker_point_count": native_point_count,
                "tracker_3h_point_count": tracker_3h_metrics["point_count"],
                "tracker_6h_point_count": tracker_6h_metrics["point_count"],
                "tracker_point_reduction": point_reduction,
                "tracker_point_reduction_percent": point_reduction_percent,
                "mean_track_error_3h_km": tracker_3h_metrics[
                    "mean_track_error_km"
                ],
                "mean_track_error_6h_km": tracker_6h_metrics[
                    "mean_track_error_km"
                ],
                "mean_track_error_change_km": mean_error_change,
                "rmse_track_error_3h_km": tracker_3h_metrics[
                    "rmse_track_error_km"
                ],
                "rmse_track_error_6h_km": tracker_6h_metrics[
                    "rmse_track_error_km"
                ],
                "rmse_track_error_change_km": rmse_error_change,
                "minimum_pressure_3h_pa": tracker_3h_metrics[
                    "minimum_pressure_pa"
                ],
                "minimum_pressure_6h_pa": tracker_6h_metrics[
                    "minimum_pressure_pa"
                ],
                "minimum_pressure_3h_lead_hours": tracker_3h_metrics[
                    "minimum_pressure_lead_hours"
                ],
                "minimum_pressure_6h_lead_hours": tracker_6h_metrics[
                    "minimum_pressure_lead_hours"
                ],
                "maximum_wind_3h_ms": tracker_3h_metrics[
                    "maximum_wind_ms"
                ],
                "maximum_wind_6h_ms": tracker_6h_metrics[
                    "maximum_wind_ms"
                ],
                "maximum_wind_3h_lead_hours": tracker_3h_metrics[
                    "maximum_wind_lead_hours"
                ],
                "maximum_wind_6h_lead_hours": tracker_6h_metrics[
                    "maximum_wind_lead_hours"
                ],
                "intensity_extrema_identical": intensity_extrema_identical,
                "first_organized_3h_lead_hours": first_organized_3h,
                "first_organized_6h_lead_hours": first_organized_6h,
                "organization_timing_difference_hours": timing_change,
                "first_zeta_threshold_3h_lead_hours": pre3_metrics[
                    "first_zeta_threshold_lead_hours"
                ],
                "first_zeta_threshold_6h_lead_hours": pre6_metrics[
                    "first_zeta_threshold_lead_hours"
                ],
                "first_pressure_deficit_threshold_3h_lead_hours": (
                    pre3_metrics[
                        "first_pressure_deficit_threshold_lead_hours"
                    ]
                ),
                "first_pressure_deficit_threshold_6h_lead_hours": (
                    pre6_metrics[
                        "first_pressure_deficit_threshold_lead_hours"
                    ]
                ),
                "common_pregenesis_point_count": (
                    common_pregenesis_point_count
                ),
                "all_common_pregenesis_identical": (
                    all_common_pregenesis_identical
                ),
                "interpretation": (
                    "temporal-sampling sensitivity; same Pangu3 "
                    "forecast trajectory, different retained output cadence"
                ),
            }
        ]
    )

    summary_output = output_dir / "sensitivity_summary.csv"
    summary.to_csv(summary_output, index=False)

    print()
    print("=" * 78)
    print("RESULT")
    print("=" * 78)
    print("Native tracker points       :", native_point_count)
    print(
        "Tracker 3 h / 6 h points    :",
        tracker_3h_metrics["point_count"],
        "/",
        tracker_6h_metrics["point_count"],
    )
    print(
        "Point reduction             :",
        point_reduction,
        f"({point_reduction_percent:.1f}%)",
    )
    print(
        "Mean track error 3 h / 6 h  :",
        f"{tracker_3h_metrics['mean_track_error_km']:.3f}",
        "/",
        f"{tracker_6h_metrics['mean_track_error_km']:.3f}",
        "km",
    )
    print(
        "RMSE track error 3 h / 6 h  :",
        f"{tracker_3h_metrics['rmse_track_error_km']:.3f}",
        "/",
        f"{tracker_6h_metrics['rmse_track_error_km']:.3f}",
        "km",
    )
    print(
        "First organized 3 h / 6 h  :",
        first_organized_3h,
        "/",
        first_organized_6h,
        "h",
    )
    print("Organization timing change  :", timing_change, "h")
    print(
        "Intensity extrema identical :",
        intensity_extrema_identical,
    )
    print(
        "Common pre-genesis identical:",
        all_common_pregenesis_identical,
    )
    print()
    print("Products:")
    print(" ", tracker_output)
    print(" ", pregenesis_output)
    print(" ", identity_output)
    print(" ", summary_output)


if __name__ == "__main__":
    main()
