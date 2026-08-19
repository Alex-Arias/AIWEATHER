from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from aiweather.verification.batch import (
    TCVerificationBatchResult,
    TCVerificationCase,
    run_tc_verification_batch,
)
from aiweather.verification.comparison import (
    TrackVerification,
)
from aiweather.verification.pipeline import (
    TCVerificationPipelineResult,
)
from aiweather.verification.workflow import (
    VerificationWorkflowResult,
)

def make_track_verification(
    name: str,
    *,
    track_errors: list[float],
    pressure_errors: list[float],
    wind_errors: list[float],
) -> TrackVerification:
    count = len(
        track_errors
    )

    valid_times = pd.date_range(
        start="2026-07-24T00:00:00",
        periods=count,
        freq="6h",
    )

    table = pd.DataFrame(
        {
            "valid_time": valid_times,
            "track_error_km": track_errors,
            "pressure_error_pa": pressure_errors,
            "wind_error_ms": wind_errors,
        }
    )

    return TrackVerification(
        forecast_name=name,
        observation_name="ibtracs_test",
        table=table,
    )

def make_pipeline_result():
    native = make_track_verification(
        "native",
        track_errors=[
            100.0,
            200.0,
            300.0,
        ],
        pressure_errors=[
            1000.0,
            -2000.0,
            1500.0,
        ],
        wind_errors=[
            5.0,
            -10.0,
            7.0,
        ],
    )

    verification = VerificationWorkflowResult(
        observations=[],
        native=native,
        wuduan=None,
        vitart=None,
    )

    return TCVerificationPipelineResult(
        tracking=SimpleNamespace(),
        verification=verification,
        output_dir=Path(
            "results/verification/test_case"
        ),
        plot_paths={},
    )


def make_case(
    *,
    case_id="case_001",
):
    return TCVerificationCase(
        forecast_path=(
            "outputs/graphcast/"
            "20260724T000000/"
            "forecast.zarr"
        ),
        sid="2026204N08267",
        lat_min=5.0,
        lat_max=35.0,
        lon_min=-130.0,
        lon_max=-90.0,
        case_id=case_id,
    )


def test_run_tc_verification_batch_single_case(
    monkeypatch,
):
    case = make_case()

    pipeline_result = (
        make_pipeline_result()
    )

    pipeline_calls = []

    def fake_pipeline(
        forecast_path,
        **kwargs,
    ):
        pipeline_calls.append(
            (
                forecast_path,
                kwargs,
            )
        )

        return pipeline_result

    monkeypatch.setattr(
        "aiweather.verification.batch."
        "run_tc_verification_pipeline",
        fake_pipeline,
    )

    forecast = SimpleNamespace(
        metadata=SimpleNamespace(
            model_name="graphcast",
            model_version="unknown",
            backend="earth2studio",
            forecast_id=(
                "graphcast_gfs_"
                "20260724T000000_240h"
            ),
            initialization_time=datetime(
                2026,
                7,
                24,
                0,
                0,
            ),
        )
    )

    monkeypatch.setattr(
        "aiweather.verification.batch."
        "open_forecast",
        lambda path: forecast,
    )

    result = run_tc_verification_batch(
        [
            case,
        ],
        device="cuda",
        minimum_overlap=3,
        generate_plots=True,
        field_lead_times=[
            54,
            78,
            96,
            120,
        ],
        ibtracs_path="ibtracs.csv",
        ibtracs_basin="EP",
        ibtracs_cache_dir=(
            "data/verification/ibtracs"
        ),
        ibtracs_max_age_hours=72.0,
        ibtracs_force_update=True,
    )

    assert isinstance(
        result,
        TCVerificationBatchResult,
    )

    assert result.cases == [
        case,
    ]

    assert result.results == [
        pipeline_result,
    ]

    assert len(
        pipeline_calls
    ) == 1

    forecast_path, kwargs = (
        pipeline_calls[0]
    )

    assert forecast_path == (
        case.forecast_path
    )

    assert kwargs == {
        "sid": case.sid,
        "lat_min": case.lat_min,
        "lat_max": case.lat_max,
        "lon_min": case.lon_min,
        "lon_max": case.lon_max,
        "device": "cuda",
        "minimum_overlap": 3,
        "generate_plots": True,
        "field_lead_times": [
            54,
            78,
            96,
            120,
        ],
        "ibtracs_path":
            "ibtracs.csv",
        "ibtracs_basin":
            "EP",
        "ibtracs_cache_dir":
            "data/verification/ibtracs",
        "ibtracs_max_age_hours":
            72.0,
        "ibtracs_force_update":
            True,
    }


def test_batch_summary_contains_case_metadata(
    monkeypatch,
):
    case = make_case(
        case_id="genevieve_20260724",
    )

    pipeline_result = (
        make_pipeline_result()
    )

    monkeypatch.setattr(
        "aiweather.verification.batch."
        "run_tc_verification_pipeline",
        lambda *args, **kwargs:
            pipeline_result,
    )

    forecast = SimpleNamespace(
        metadata=SimpleNamespace(
            model_name="graphcast",
            model_version="unknown",
            backend="earth2studio",
            forecast_id=(
                "graphcast_gfs_"
                "20260724T000000_240h"
            ),
            initialization_time=datetime(
                2026,
                7,
                24,
                0,
                0,
            ),
        )
    )

    monkeypatch.setattr(
        "aiweather.verification.batch."
        "open_forecast",
        lambda path: forecast,
    )

    result = run_tc_verification_batch(
        [
            case,
        ]
    )

    summary = result.summary

    assert not summary.empty

    assert set(
        summary["coverage"]
    ) == {
        "full",
        "common",
    }

    assert set(
        summary["tracker"]
    ) == {
        "native",
    }

    row = summary.iloc[0]

    assert row[
        "case_id"
    ] == "genevieve_20260724"

    assert row[
        "model_name"
    ] == "graphcast"

    assert row[
        "model_version"
    ] == "unknown"

    assert row[
        "backend"
    ] == "earth2studio"

    assert row[
        "forecast_id"
    ] == (
        "graphcast_gfs_"
        "20260724T000000_240h"
    )

    assert row[
        "forecast_path"
    ] == str(
        case.forecast_path
    )

    assert row[
        "sid"
    ] == "2026204N08267"

    assert row[
        "initialization_time"
    ] == datetime(
        2026,
        7,
        24,
        0,
        0,
    )


def test_batch_summary_contains_metrics(
    monkeypatch,
):
    case = make_case()

    pipeline_result = (
        make_pipeline_result()
    )

    monkeypatch.setattr(
        "aiweather.verification.batch."
        "run_tc_verification_pipeline",
        lambda *args, **kwargs:
            pipeline_result,
    )

    forecast = SimpleNamespace(
        metadata=SimpleNamespace(
            model_name="graphcast",
            model_version="unknown",
            backend="earth2studio",
            forecast_id="forecast-id",
            initialization_time=datetime(
                2026,
                7,
                24,
            ),
        )
    )

    monkeypatch.setattr(
        "aiweather.verification.batch."
        "open_forecast",
        lambda path: forecast,
    )

    result = run_tc_verification_batch(
        [
            case,
        ]
    )

    full = result.summary[
        result.summary["coverage"]
        == "full"
    ].iloc[0]

    assert full[
        "overlap_count"
    ] == 3

    assert full[
        "mean_track_error_km"
    ] == pytest.approx(
        200.0
    )

    assert full[
        "rmse_track_error_km"
    ] == pytest.approx(
        (
            (
                100.0 ** 2
                + 200.0 ** 2
                + 300.0 ** 2
            )
            / 3.0
        ) ** 0.5
    )

    assert full[
        "median_track_error_km"
    ] == pytest.approx(
        200.0
    )

    assert full[
        "maximum_track_error_km"
    ] == pytest.approx(
        300.0
    )

    assert full[
        "pressure_mae_pa"
    ] == pytest.approx(
        1500.0
    )

    assert full[
        "wind_mae_ms"
    ] == pytest.approx(
        (
            5.0
            + 10.0
            + 7.0
        )
        / 3.0
    )


def test_run_tc_verification_batch_multiple_cases(
    monkeypatch,
):
    cases = [
        make_case(
            case_id="case_a"
        ),
        TCVerificationCase(
            forecast_path=(
                "outputs/graphcast/"
                "20260725T000000/"
                "forecast.zarr"
            ),
            sid="2026205N10266",
            lat_min=5.0,
            lat_max=35.0,
            lon_min=-130.0,
            lon_max=-90.0,
            case_id="case_b",
        ),
    ]

    pipeline_result = (
        make_pipeline_result()
    )

    calls = []

    def fake_pipeline(
        forecast_path,
        **kwargs,
    ):
        calls.append(
            forecast_path
        )

        return pipeline_result

    monkeypatch.setattr(
        "aiweather.verification.batch."
        "run_tc_verification_pipeline",
        fake_pipeline,
    )

    def fake_open_forecast(
        path,
    ):
        init = (
            datetime(
                2026,
                7,
                24,
            )
            if "20260724" in str(path)
            else datetime(
                2026,
                7,
                25,
            )
        )

        return SimpleNamespace(
            metadata=SimpleNamespace(
                model_name="graphcast",
                model_version="unknown",
                backend="earth2studio",
                forecast_id=str(path),
                initialization_time=init,
            )
        )

    monkeypatch.setattr(
        "aiweather.verification.batch."
        "open_forecast",
        fake_open_forecast,
    )

    result = run_tc_verification_batch(
        cases
    )

    assert len(
        result.results
    ) == 2

    assert calls == [
        cases[0].forecast_path,
        cases[1].forecast_path,
    ]

    assert set(
        result.summary["case_id"]
    ) == {
        "case_a",
        "case_b",
    }


def test_run_tc_verification_batch_invalid_cases_type():
    with pytest.raises(
        TypeError,
        match="cases must be a list",
    ):
        run_tc_verification_batch(
            (
                make_case(),
            )
        )


def test_run_tc_verification_batch_empty_cases():
    with pytest.raises(
        ValueError,
        match="cases cannot be empty",
    ):
        run_tc_verification_batch(
            []
        )


def test_batch_case_invalid_type():
    with pytest.raises(
        TypeError,
        match="TCVerificationCase",
    ):
        run_tc_verification_batch(
            [
                "not-a-case",
            ]
        )


def test_batch_case_empty_sid():
    case = make_case()
    case.sid = "   "

    with pytest.raises(
        ValueError,
        match="case.sid cannot be empty",
    ):
        run_tc_verification_batch(
            [
                case,
            ]
        )


def test_batch_case_invalid_latitude_bounds():
    case = make_case()
    case.lat_min = 35.0
    case.lat_max = 5.0

    with pytest.raises(
        ValueError,
        match="case.lat_min",
    ):
        run_tc_verification_batch(
            [
                case,
            ]
        )


def test_batch_case_invalid_longitude_bounds():
    case = make_case()
    case.lon_min = -90.0
    case.lon_max = -130.0

    with pytest.raises(
        ValueError,
        match="case.lon_min",
    ):
        run_tc_verification_batch(
            [
                case,
            ]
        )


def test_batch_case_invalid_case_id_type():
    case = make_case()
    case.case_id = 123

    with pytest.raises(
        TypeError,
        match="case.case_id",
    ):
        run_tc_verification_batch(
            [
                case,
            ]
        )


def test_batch_case_empty_case_id():
    case = make_case()
    case.case_id = "   "

    with pytest.raises(
        ValueError,
        match="case.case_id cannot be empty",
    ):
        run_tc_verification_batch(
            [
                case,
            ]
        )

def test_batch_without_output_dir_does_not_export(
    monkeypatch,
):
    case = make_case()

    pipeline_result = (
        make_pipeline_result()
    )

    monkeypatch.setattr(
        "aiweather.verification.batch."
        "run_tc_verification_pipeline",
        lambda *args, **kwargs:
            pipeline_result,
    )

    forecast = SimpleNamespace(
        metadata=SimpleNamespace(
            model_name="graphcast",
            model_version="unknown",
            backend="earth2studio",
            forecast_id="forecast-id",
            initialization_time=datetime(
                2026,
                7,
                24,
            ),
        )
    )

    monkeypatch.setattr(
        "aiweather.verification.batch."
        "open_forecast",
        lambda path: forecast,
    )

    result = run_tc_verification_batch(
        [
            case,
        ]
    )

    assert result.output_dir is None
    assert result.summary_path is None
    assert result.aggregate_path is None
    assert result.lead_time_path is None

def test_batch_summary_export(
    monkeypatch,
    tmp_path,
):
    case = make_case()

    pipeline_result = (
        make_pipeline_result()
    )

    monkeypatch.setattr(
        "aiweather.verification.batch."
        "run_tc_verification_pipeline",
        lambda *args, **kwargs:
            pipeline_result,
    )

    forecast = SimpleNamespace(
        metadata=SimpleNamespace(
            model_name="graphcast",
            model_version="unknown",
            backend="earth2studio",
            forecast_id="forecast-id",
            initialization_time=datetime(
                2026,
                7,
                24,
            ),
        )
    )

    monkeypatch.setattr(
        "aiweather.verification.batch."
        "open_forecast",
        lambda path: forecast,
    )

    output_dir = (
        tmp_path
        / "batch"
    )

    result = run_tc_verification_batch(
        [
            case,
        ],
        output_dir=output_dir,
    )

    expected_summary = (
        output_dir
        / "batch_summary.csv"
    )

    expected_aggregate = (
        output_dir
        / "batch_aggregate.csv"
    )

    expected_lead_time = (
        output_dir
        / "batch_lead_time.csv"
    )

    assert result.output_dir == (
        output_dir
    )

    assert result.summary_path == (
        expected_summary
    )

    assert result.aggregate_path == (
        expected_aggregate
    )

    assert result.lead_time_path == (
        expected_lead_time
    )

    assert expected_summary.exists()
    assert expected_aggregate.exists()
    assert expected_lead_time.exists()

    exported_summary = pd.read_csv(
        expected_summary
    )

    exported_aggregate = pd.read_csv(
        expected_aggregate
    )

    exported_lead_time = pd.read_csv(
        expected_lead_time
    )

    assert len(exported_summary) == len(
        result.summary
    )

    assert list(
        exported_summary["tracker"]
    ) == list(
        result.summary["tracker"]
    )

    assert list(
        exported_summary["coverage"]
    ) == list(
        result.summary["coverage"]
    )

    assert list(
        exported_summary["sid"]
    ) == list(
        result.summary["sid"]
    )

    assert list(
        exported_aggregate.columns
    ) == list(
        result.aggregate.columns
    )

    assert len(
        exported_aggregate
    ) == len(
        result.aggregate
    )

    assert list(
        exported_lead_time.columns
    ) == list(
        result.lead_time.columns
    )

    assert len(
        exported_lead_time
    ) == len(
        result.lead_time
    )

def test_batch_summary_contains_qc_columns(
    monkeypatch,
):
    case = make_case()

    pipeline_result = (
        make_pipeline_result()
    )

    monkeypatch.setattr(
        "aiweather.verification.batch."
        "run_tc_verification_pipeline",
        lambda *args, **kwargs:
            pipeline_result,
    )

    forecast = SimpleNamespace(
        metadata=SimpleNamespace(
            model_name="graphcast",
            model_version="unknown",
            backend="earth2studio",
            forecast_id="forecast-id",
            initialization_time=datetime(
                2026,
                7,
                24,
            ),
        )
    )

    monkeypatch.setattr(
        "aiweather.verification.batch."
        "open_forecast",
        lambda path: forecast,
    )

    result = run_tc_verification_batch(
        [
            case,
        ]
    )

    assert {
        "qc_status",
        "qc_reason",
        "initial_separation_km",
        "valid_for_aggregation",
    }.issubset(
        result.summary.columns
    )


def test_batch_qc_pass(
    monkeypatch,
):
    case = make_case()

    pipeline_result = (
        make_pipeline_result()
    )

    monkeypatch.setattr(
        "aiweather.verification.batch."
        "run_tc_verification_pipeline",
        lambda *args, **kwargs:
            pipeline_result,
    )

    forecast = SimpleNamespace(
        metadata=SimpleNamespace(
            model_name="graphcast",
            model_version="unknown",
            backend="earth2studio",
            forecast_id="forecast-id",
            initialization_time=datetime(
                2026,
                7,
                24,
            ),
        )
    )

    monkeypatch.setattr(
        "aiweather.verification.batch."
        "open_forecast",
        lambda path: forecast,
    )

    result = run_tc_verification_batch(
        [
            case,
        ]
    )

    full = result.summary[
        result.summary["coverage"]
        == "full"
    ].iloc[0]

    assert full[
        "qc_status"
    ] == "limited"

    assert full[
        "qc_reason"
    ] == "short_overlap"

    assert (
        full[
            "valid_for_aggregation"
        ]
        is False
        or full[
            "valid_for_aggregation"
        ] == False
    )

def test_batch_qc_fail_initial_association(
    monkeypatch,
):
    case = make_case()

    bad = make_track_verification(
        "native",
        track_errors=[
            10500.0,
            10600.0,
            10700.0,
            10800.0,
            10900.0,
            11000.0,
        ],
        pressure_errors=[
            100.0,
            100.0,
            100.0,
            100.0,
            100.0,
            100.0,
        ],
        wind_errors=[
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
        ],
    )

    verification = VerificationWorkflowResult(
        observations=[],
        native=bad,
        wuduan=None,
        vitart=None,
    )

    pipeline_result = (
        TCVerificationPipelineResult(
            tracking=SimpleNamespace(),
            verification=verification,
            output_dir=None,
            plot_paths={},
        )
    )

    monkeypatch.setattr(
        "aiweather.verification.batch."
        "run_tc_verification_pipeline",
        lambda *args, **kwargs:
            pipeline_result,
    )

    forecast = SimpleNamespace(
        metadata=SimpleNamespace(
            model_name="graphcast",
            model_version="unknown",
            backend="earth2studio",
            forecast_id="forecast-id",
            initialization_time=datetime(
                2026,
                7,
                24,
            ),
        )
    )

    monkeypatch.setattr(
        "aiweather.verification.batch."
        "open_forecast",
        lambda path: forecast,
    )

    result = run_tc_verification_batch(
        [
            case,
        ]
    )

    full = result.summary[
        result.summary["coverage"]
        == "full"
    ].iloc[0]

    assert full[
        "qc_status"
    ] == "fail"

    assert full[
        "qc_reason"
    ] == "initial_association"

    assert full[
        "initial_separation_km"
    ] == pytest.approx(
        10500.0
    )

    assert (
        full[
            "valid_for_aggregation"
        ]
        is False
        or full[
            "valid_for_aggregation"
        ] == False
    )

def test_batch_result_contains_aggregate(
    monkeypatch,
):
    case = make_case()

    valid = make_track_verification(
        "native",
        track_errors=[
            100.0,
            120.0,
            140.0,
            160.0,
            180.0,
            200.0,
        ],
        pressure_errors=[
            1000.0,
            1100.0,
            1200.0,
            1300.0,
            1400.0,
            1500.0,
        ],
        wind_errors=[
            10.0,
            11.0,
            12.0,
            13.0,
            14.0,
            15.0,
        ],
    )

    valid.table[
        "lead_time_hours"
    ] = [
        0.0,
        6.0,
        12.0,
        18.0,
        24.0,
        30.0,
    ]

    verification = VerificationWorkflowResult(
        observations=[],
        native=valid,
        wuduan=None,
        vitart=None,
    )

    pipeline_result = (
        TCVerificationPipelineResult(
            tracking=SimpleNamespace(),
            verification=verification,
            output_dir=None,
            plot_paths={},
        )
    )

    monkeypatch.setattr(
        "aiweather.verification.batch."
        "run_tc_verification_pipeline",
        lambda *args, **kwargs:
            pipeline_result,
    )

    forecast = SimpleNamespace(
        metadata=SimpleNamespace(
            model_name="graphcast",
            model_version="unknown",
            backend="earth2studio",
            forecast_id="forecast-id",
            initialization_time=datetime(
                2026,
                7,
                24,
            ),
        )
    )

    monkeypatch.setattr(
        "aiweather.verification.batch."
        "open_forecast",
        lambda path: forecast,
    )

    result = run_tc_verification_batch(
        [
            case,
        ]
    )

    assert not result.aggregate.empty

    full = result.aggregate[
        result.aggregate["coverage"]
        == "full"
    ].iloc[0]

    assert full[
        "case_count"
    ] == 1

    assert full[
        "mean_case_track_error_km"
    ] == pytest.approx(
        150.0
    )

def test_batch_aggregate_excludes_invalid_qc(
    monkeypatch,
):
    case = make_case()

    invalid = make_track_verification(
        "native",
        track_errors=[
            10000.0,
            10100.0,
            10200.0,
            10300.0,
            10400.0,
            10500.0,
        ],
        pressure_errors=[
            100.0,
        ] * 6,
        wind_errors=[
            1.0,
        ] * 6,
    )

    verification = VerificationWorkflowResult(
        observations=[],
        native=invalid,
        wuduan=None,
        vitart=None,
    )

    pipeline_result = (
        TCVerificationPipelineResult(
            tracking=SimpleNamespace(),
            verification=verification,
            output_dir=None,
            plot_paths={},
        )
    )

    monkeypatch.setattr(
        "aiweather.verification.batch."
        "run_tc_verification_pipeline",
        lambda *args, **kwargs:
            pipeline_result,
    )

    forecast = SimpleNamespace(
        metadata=SimpleNamespace(
            model_name="graphcast",
            model_version="unknown",
            backend="earth2studio",
            forecast_id="forecast-id",
            initialization_time=datetime(
                2026,
                7,
                24,
            ),
        )
    )

    monkeypatch.setattr(
        "aiweather.verification.batch."
        "open_forecast",
        lambda path: forecast,
    )

    result = run_tc_verification_batch(
        [
            case,
        ]
    )

    assert not result.summary.empty

    assert (
        result.summary[
            "valid_for_aggregation"
        ].eq(False).all()
    )

    assert result.aggregate.empty

def test_batch_result_contains_lead_time(
    monkeypatch,
):
    case = make_case()

    valid = make_track_verification(
        "native",
        track_errors=[
            100.0,
            120.0,
            140.0,
            160.0,
            180.0,
            200.0,
        ],
        pressure_errors=[
            1000.0,
            1100.0,
            1200.0,
            1300.0,
            1400.0,
            1500.0,
        ],
        wind_errors=[
            10.0,
            11.0,
            12.0,
            13.0,
            14.0,
            15.0,
        ],
    )

    valid.table[
        "lead_time_hours"
    ] = [
        0.0,
        6.0,
        12.0,
        18.0,
        24.0,
        30.0,
    ]

    verification = VerificationWorkflowResult(
        observations=[],
        native=valid,
        wuduan=None,
        vitart=None,
    )

    pipeline_result = (
        TCVerificationPipelineResult(
            tracking=SimpleNamespace(),
            verification=verification,
            output_dir=None,
            plot_paths={},
        )
    )

    monkeypatch.setattr(
        "aiweather.verification.batch."
        "run_tc_verification_pipeline",
        lambda *args, **kwargs:
            pipeline_result,
    )

    forecast = SimpleNamespace(
        metadata=SimpleNamespace(
            model_name="graphcast",
            model_version="unknown",
            backend="earth2studio",
            forecast_id="forecast-id",
            initialization_time=datetime(
                2026,
                7,
                24,
            ),
        )
    )

    monkeypatch.setattr(
        "aiweather.verification.batch."
        "open_forecast",
        lambda path: forecast,
    )

    result = run_tc_verification_batch(
        [
            case,
        ]
    )

    assert not result.lead_time.empty

    assert set(
        result.lead_time[
            "lead_time_bin"
        ]
    ) == {
        "0-24h",
        "24-48h",
    }

    first = result.lead_time[
        result.lead_time[
            "lead_time_bin"
        ].eq(
            "0-24h"
        )
    ].iloc[0]

    assert first[
        "case_count"
    ] == 1

    assert first[
        "point_count"
    ] == 4
