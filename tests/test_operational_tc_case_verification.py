from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "verify_operational_tc_case.py"
)

SPEC = importlib.util.spec_from_file_location(
    "verify_operational_tc_case",
    SCRIPT_PATH,
)

MODULE = importlib.util.module_from_spec(
    SPEC
)

assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def _verification_table(
    valid_times: list[str],
    lead_times: list[int],
    track_errors: list[float],
) -> pd.DataFrame:
    n = len(valid_times)

    return pd.DataFrame(
        {
            "lead_time_hours": lead_times,
            "valid_time": valid_times,
            "track_error_km": track_errors,
            "pressure_error_pa": np.arange(
                n,
                dtype=float,
            ),
            "wind_error_ms": np.arange(
                n,
                dtype=float,
            ),
        }
    )


def _write_track(
    input_dir: Path,
    model: str,
) -> Path:
    model_dir = input_dir / model
    model_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    track_path = (
        model_dir
        / f"{model}_track.csv"
    )

    pd.DataFrame(
        {
            "lead_time_hours": [0],
            "valid_time": [
                "2026-09-10T12:00:00"
            ],
            "latitude": [16.5],
            "longitude": [-118.6],
        }
    ).to_csv(
        track_path,
        index=False,
    )

    return track_path


def test_main_preserves_native_cadence_and_builds_common_times(
    tmp_path,
    monkeypatch,
):
    input_dir = tmp_path / "operational"
    output_dir = tmp_path / "verification"

    _write_track(
        input_dir,
        "model6",
    )
    _write_track(
        input_dir,
        "model3",
    )

    model6_table = _verification_table(
        [
            "2026-09-10T12:00:00",
            "2026-09-10T18:00:00",
            "2026-09-11T00:00:00",
        ],
        [0, 6, 12],
        [10.0, 20.0, 30.0],
    )

    model3_table = _verification_table(
        [
            "2026-09-10T12:00:00",
            "2026-09-10T15:00:00",
            "2026-09-10T18:00:00",
            "2026-09-10T21:00:00",
            "2026-09-11T00:00:00",
        ],
        [0, 3, 6, 9, 12],
        [
            10.0,
            999.0,
            20.0,
            999.0,
            30.0,
        ],
    )

    tables = {
        "model6": model6_table,
        "model3": model3_table,
    }

    def fake_verify(
        track_path,
        ibtracs_path,
        *,
        sid,
        initialization_time,
        forecast_name,
        observation_name,
    ):
        table = tables[
            forecast_name
        ].copy()

        track_error = table[
            "track_error_km"
        ]

        return SimpleNamespace(
            table=table,
            overlap_count=len(table),
            mean_track_error_km=(
                track_error.mean()
            ),
            rmse_track_error_km=float(
                np.sqrt(
                    np.mean(
                        np.square(
                            track_error
                        )
                    )
                )
            ),
            median_track_error_km=(
                track_error.median()
            ),
            maximum_track_error_km=(
                track_error.max()
            ),
        )

    monkeypatch.setattr(
        MODULE,
        "verify_operational_track_csv",
        fake_verify,
    )

    monkeypatch.setattr(
        "sys.argv",
        [
            "verify_operational_tc_case.py",
            "--input-dir",
            str(input_dir),
            "--output-dir",
            str(output_dir),
            "--ibtracs",
            str(tmp_path / "ibtracs.csv"),
            "--sid",
            "TESTSID",
            "--initialization-time",
            "2026-09-10T12:00:00",
        ],
    )

    MODULE.main()

    native3 = pd.read_csv(
        output_dir
        / "model3_verification.csv"
    )
    native6 = pd.read_csv(
        output_dir
        / "model6_verification.csv"
    )

    assert native3[
        "lead_time_hours"
    ].tolist() == [
        0,
        3,
        6,
        9,
        12,
    ]

    assert native6[
        "lead_time_hours"
    ].tolist() == [
        0,
        6,
        12,
    ]

    common3 = pd.read_csv(
        output_dir
        / "model3_common_verification.csv"
    )
    common6 = pd.read_csv(
        output_dir
        / "model6_common_verification.csv"
    )

    expected_common_leads = [
        0,
        6,
        12,
    ]

    assert common3[
        "lead_time_hours"
    ].tolist() == expected_common_leads

    assert common6[
        "lead_time_hours"
    ].tolist() == expected_common_leads

    assert common3[
        "track_error_km"
    ].tolist() == [
        10.0,
        20.0,
        30.0,
    ]

    assert common6[
        "track_error_km"
    ].tolist() == [
        10.0,
        20.0,
        30.0,
    ]

    common_summary = pd.read_csv(
        output_dir
        / "common_verification_summary.csv"
    ).set_index(
        "model"
    )

    assert (
        common_summary.loc[
            "model3",
            "overlap_count",
        ]
        == 3
    )

    assert (
        common_summary.loc[
            "model6",
            "overlap_count",
        ]
        == 3
    )

    assert (
        common_summary.loc[
            "model3",
            "mean_track_error_km",
        ]
        == pytest.approx(
            common_summary.loc[
                "model6",
                "mean_track_error_km",
            ]
        )
    )

    assert (
        common_summary.loc[
            "model3",
            "rmse_track_error_km",
        ]
        == pytest.approx(
            common_summary.loc[
                "model6",
                "rmse_track_error_km",
            ]
        )
    )

    assert (
        output_dir
        / "verification_summary.csv"
    ).exists()

    assert (
        output_dir
        / "lead_time_summary.csv"
    ).exists()

    assert (
        output_dir
        / "common_lead_time_summary.csv"
    ).exists()


def test_main_rejects_no_common_valid_times(
    tmp_path,
    monkeypatch,
):
    input_dir = tmp_path / "operational"
    output_dir = tmp_path / "verification"

    _write_track(
        input_dir,
        "model_a",
    )
    _write_track(
        input_dir,
        "model_b",
    )

    tables = {
        "model_a": _verification_table(
            [
                "2026-09-10T12:00:00",
                "2026-09-10T18:00:00",
            ],
            [0, 6],
            [10.0, 20.0],
        ),
        "model_b": _verification_table(
            [
                "2026-09-10T15:00:00",
                "2026-09-10T21:00:00",
            ],
            [3, 9],
            [30.0, 40.0],
        ),
    }

    def fake_verify(
        track_path,
        ibtracs_path,
        *,
        sid,
        initialization_time,
        forecast_name,
        observation_name,
    ):
        table = tables[
            forecast_name
        ].copy()

        track_error = table[
            "track_error_km"
        ]

        return SimpleNamespace(
            table=table,
            overlap_count=len(table),
            mean_track_error_km=(
                track_error.mean()
            ),
            rmse_track_error_km=float(
                np.sqrt(
                    np.mean(
                        np.square(
                            track_error
                        )
                    )
                )
            ),
            median_track_error_km=(
                track_error.median()
            ),
            maximum_track_error_km=(
                track_error.max()
            ),
        )

    monkeypatch.setattr(
        MODULE,
        "verify_operational_track_csv",
        fake_verify,
    )

    monkeypatch.setattr(
        "sys.argv",
        [
            "verify_operational_tc_case.py",
            "--input-dir",
            str(input_dir),
            "--output-dir",
            str(output_dir),
            "--ibtracs",
            str(tmp_path / "ibtracs.csv"),
            "--sid",
            "TESTSID",
            "--initialization-time",
            "2026-09-10T12:00:00",
        ],
    )

    MODULE.main()

    summary = pd.read_csv(
        output_dir / "verification_summary.csv"
    )
    common_summary = pd.read_csv(
        output_dir / "common_verification_summary.csv"
    )
    common_lead_summary = pd.read_csv(
        output_dir / "common_lead_time_summary.csv"
    )

    assert len(summary) == 2
    assert set(summary["tracker"]) == {"native"}

    assert common_summary.empty
    assert list(common_summary.columns) == [
        "model",
        "tracker",
        "forecast_track",
        "sid",
        "initialization_time",
        "overlap_count",
        "first_lead_time_hours",
        "last_lead_time_hours",
        "mean_track_error_km",
        "rmse_track_error_km",
        "median_track_error_km",
        "maximum_track_error_km",
        "pressure_bias_pa",
        "pressure_mae_pa",
        "pressure_rmse_pa",
        "wind_bias_ms",
        "wind_mae_ms",
        "wind_rmse_ms",
    ]

    assert common_lead_summary.empty
    assert "tracker" in common_lead_summary.columns


def test_main_separates_trackers_and_common_times_by_tracker(
    tmp_path,
    monkeypatch,
):
    """Keep model/tracker products distinct and compare within tracker."""

    input_dir = tmp_path / "operational"
    output_dir = tmp_path / "verification"

    for model in ("model_a", "model_b"):
        model_dir = input_dir / model
        model_dir.mkdir(parents=True)

        for tracker in ("native", "wuduan"):
            if tracker == "native":
                name = f"{model}_track.csv"
            else:
                name = f"{model}_{tracker}_track.csv"

            pd.DataFrame(
                {
                    "lead_time_hours": [0],
                    "valid_time": ["2026-09-10T12:00:00"],
                    "latitude": [16.5],
                    "longitude": [-118.6],
                }
            ).to_csv(
                model_dir / name,
                index=False,
            )

    tables = {
        ("model_a", "native"): _verification_table(
            [
                "2026-09-10T12:00:00",
                "2026-09-10T18:00:00",
            ],
            [0, 6],
            [10.0, 20.0],
        ),
        ("model_b", "native"): _verification_table(
            [
                "2026-09-10T12:00:00",
                "2026-09-10T18:00:00",
            ],
            [0, 6],
            [30.0, 40.0],
        ),
        ("model_a", "wuduan"): _verification_table(
            [
                "2026-09-10T15:00:00",
                "2026-09-10T21:00:00",
            ],
            [3, 9],
            [50.0, 60.0],
        ),
        ("model_b", "wuduan"): _verification_table(
            [
                "2026-09-10T15:00:00",
                "2026-09-10T21:00:00",
            ],
            [3, 9],
            [70.0, 80.0],
        ),
    }

    def fake_verify(
        track_path,
        ibtracs_path,
        *,
        sid,
        initialization_time,
        forecast_name,
        observation_name,
    ):
        name = Path(track_path).name

        tracker = (
            "wuduan"
            if "_wuduan_track.csv" in name
            else "native"
        )

        table = tables[
            (forecast_name, tracker)
        ].copy()

        return SimpleNamespace(
            table=table
        )

    monkeypatch.setattr(
        MODULE,
        "verify_operational_track_csv",
        fake_verify,
    )

    monkeypatch.setattr(
        "sys.argv",
        [
            "verify_operational_tc_case.py",
            "--input-dir",
            str(input_dir),
            "--output-dir",
            str(output_dir),
            "--ibtracs",
            str(tmp_path / "ibtracs.csv"),
            "--sid",
            "TESTSID",
            "--initialization-time",
            "2026-09-10T12:00:00",
        ],
    )

    MODULE.main()

    summary = pd.read_csv(
        output_dir / "verification_summary.csv"
    )

    assert len(summary) == 4

    assert set(
        zip(
            summary["model"],
            summary["tracker"],
        )
    ) == {
        ("model_a", "native"),
        ("model_b", "native"),
        ("model_a", "wuduan"),
        ("model_b", "wuduan"),
    }

    for filename in (
        "model_a_native_verification.csv",
        "model_b_native_verification.csv",
        "model_a_wuduan_verification.csv",
        "model_b_wuduan_verification.csv",
        "model_a_native_common_verification.csv",
        "model_b_native_common_verification.csv",
        "model_a_wuduan_common_verification.csv",
        "model_b_wuduan_common_verification.csv",
    ):
        assert (
            output_dir / filename
        ).exists()

    native_common = pd.read_csv(
        output_dir
        / "model_a_native_common_verification.csv"
    )

    wuduan_common = pd.read_csv(
        output_dir
        / "model_a_wuduan_common_verification.csv"
    )

    assert native_common[
        "lead_time_hours"
    ].tolist() == [0, 6]

    assert wuduan_common[
        "lead_time_hours"
    ].tolist() == [3, 9]

    common_summary = pd.read_csv(
        output_dir
        / "common_verification_summary.csv"
    )

    assert set(
        common_summary["tracker"]
    ) == {
        "native",
        "wuduan",
    }

    lead_summary = pd.read_csv(
        output_dir / "lead_time_summary.csv"
    )

    assert "tracker" in lead_summary.columns

    common_lead_summary = pd.read_csv(
        output_dir
        / "common_lead_time_summary.csv"
    )

    assert "tracker" in common_lead_summary.columns
