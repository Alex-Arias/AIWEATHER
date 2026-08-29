from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "analyze_pangu_temporal_sampling_sensitivity.py"
)

SPEC = importlib.util.spec_from_file_location(
    "pangu_temporal_sampling",
    SCRIPT_PATH,
)

MODULE = importlib.util.module_from_spec(
    SPEC
)

assert SPEC.loader is not None
SPEC.loader.exec_module(
    MODULE
)


def test_tracker_metrics_3h_and_6h_sampling():
    frame = pd.DataFrame(
        {
            "lead_time_hours": [
                60,
                63,
                66,
                69,
                72,
                75,
                78,
            ],
            "track_error_km": [
                150.0,
                140.0,
                160.0,
                155.0,
                145.0,
                150.0,
                165.0,
            ],
            "forecast_pressure_pa": [
                100400.0,
                100300.0,
                100200.0,
                100100.0,
                100000.0,
                99950.0,
                99900.0,
            ],
            "forecast_wind_ms": [
                15.0,
                16.0,
                16.5,
                16.7,
                17.5,
                17.0,
                16.8,
            ],
        }
    )

    frame_6h = frame.loc[
        frame["lead_time_hours"] % 6 == 0
    ].copy()

    metrics_3h = MODULE.tracker_metrics(
        frame,
        cadence_hours=3,
    )

    metrics_6h = MODULE.tracker_metrics(
        frame_6h,
        cadence_hours=6,
    )

    assert metrics_3h["point_count"] == 7
    assert metrics_6h["point_count"] == 4
    assert metrics_3h["first_lead_hours"] == 60
    assert metrics_3h["last_lead_hours"] == 78
    assert metrics_6h["first_lead_hours"] == 60
    assert metrics_6h["last_lead_hours"] == 78
    assert metrics_3h["gap_count"] == 0
    assert metrics_6h["gap_count"] == 0


def test_pregenesis_first_organized_and_thresholds():
    frame = pd.DataFrame(
        {
            "lead_time_hours": [
                24,
                27,
                30,
                33,
                36,
            ],
            "valid_time": [
                "2026-08-22T18:00:00",
                "2026-08-22T21:00:00",
                "2026-08-23T00:00:00",
                "2026-08-23T03:00:00",
                "2026-08-23T06:00:00",
            ],
            "maximum_zeta850_1e5_s1": [
                12.0,
                15.8,
                16.2,
                16.3,
                17.0,
            ],
            "pressure_deficit_hpa": [
                1.1,
                1.1,
                1.4,
                1.6,
                2.1,
            ],
            "organized_vortex": [
                False,
                False,
                False,
                False,
                True,
            ],
        }
    )

    metrics = MODULE.pregenesis_metrics(
        frame,
        cadence_hours=3,
    )

    assert (
        metrics["first_zeta_threshold_lead_hours"]
        == 27
    )

    assert (
        metrics[
            "first_pressure_deficit_threshold_lead_hours"
        ]
        == 36
    )

    assert (
        metrics["first_organized_lead_hours"]
        == 36
    )


def test_common_pregenesis_identity():
    frame_3h = pd.DataFrame(
        {
            "lead_time_hours": [
                0,
                3,
                6,
                9,
                12,
            ],
            "minimum_mslp_hpa": [
                1010.0,
                1009.5,
                1009.0,
                1008.5,
                1008.0,
            ],
            "environment_mslp_hpa": [
                1011.0,
                1011.0,
                1011.0,
                1011.0,
                1011.0,
            ],
            "pressure_deficit_hpa": [
                1.0,
                1.5,
                2.0,
                2.5,
                3.0,
            ],
            "pressure_center_error_km": [
                100.0,
                100.0,
                100.0,
                100.0,
                100.0,
            ],
            "maximum_wind10_ms": [
                10.0,
                10.5,
                11.0,
                11.5,
                12.0,
            ],
            "maximum_zeta850_s1": [
                1.0e-4,
                1.2e-4,
                1.5e-4,
                1.7e-4,
                1.9e-4,
            ],
            "maximum_zeta850_1e5_s1": [
                10.0,
                12.0,
                15.0,
                17.0,
                19.0,
            ],
            "mean_positive_zeta850_s1": [
                5.0e-5,
                6.0e-5,
                7.0e-5,
                8.0e-5,
                9.0e-5,
            ],
            "mean_positive_zeta850_1e5_s1": [
                5.0,
                6.0,
                7.0,
                8.0,
                9.0,
            ],
            "zeta_center_error_km": [
                120.0,
                120.0,
                120.0,
                120.0,
                120.0,
            ],
            "pressure_zeta_separation_km": [
                50.0,
                50.0,
                50.0,
                50.0,
                50.0,
            ],
            "organized_vortex": [
                False,
                False,
                True,
                True,
                True,
            ],
        }
    )

    frame_6h = (
        frame_3h.loc[
            frame_3h["lead_time_hours"] % 6 == 0
        ]
        .copy()
        .reset_index(drop=True)
    )

    identity, all_identical, common_count = (
        MODULE.common_pregenesis_identity(
            frame_3h,
            frame_6h,
        )
    )

    assert common_count == 3
    assert all_identical is True
    assert identity["bitwise_identical"].all()


def test_tracker_gap_detection():
    frame = pd.DataFrame(
        {
            "lead_time_hours": [
                60,
                63,
                69,
                72,
            ],
            "track_error_km": [
                100.0,
                110.0,
                120.0,
                130.0,
            ],
            "forecast_pressure_pa": [
                100000.0,
                99900.0,
                99800.0,
                99700.0,
            ],
            "forecast_wind_ms": [
                15.0,
                16.0,
                17.0,
                18.0,
            ],
        }
    )

    metrics = MODULE.tracker_metrics(
        frame,
        cadence_hours=3,
    )

    assert metrics["gap_count"] == 1
    assert metrics["maximum_gap_hours"] == 6
