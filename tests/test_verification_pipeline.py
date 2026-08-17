from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from aiweather.tracking.workflow import (
    TrackingWorkflowResult,
)
from aiweather.verification.pipeline import (
    TCVerificationPipelineResult,
    run_tc_verification_pipeline,
)
from aiweather.verification.workflow import (
    VerificationWorkflowResult,
)


def make_tracking_result():
    return TrackingWorkflowResult(
        genesis=None,
        native_records=[],
        evaluation=None,
    )


def make_verification_result():
    return VerificationWorkflowResult(
        observations=[],
        native=None,
        wuduan=None,
        vitart=None,
    )


def test_run_tc_verification_pipeline(
    monkeypatch,
    tmp_path,
):
    calls = []

    forecast = SimpleNamespace(
        metadata=SimpleNamespace(
            initialization_time=np.datetime64(
                "2026-07-24T00:00:00"
            )
        )
    )

    tracking = make_tracking_result()
    verification = make_verification_result()

    def fake_open_forecast(path):
        calls.append(
            (
                "open_forecast",
                path,
            )
        )

        return forecast

    def fake_evaluate_forecast_trackers(
        forecast_arg,
        **kwargs,
    ):
        calls.append(
            (
                "evaluate",
                forecast_arg,
                kwargs,
            )
        )

        return tracking

    def fake_read_ibtracs_csv(
        path,
        *,
        sid,
    ):
        calls.append(
            (
                "read_ibtracs",
                path,
                sid,
            )
        )

        return [
            "best-track-point",
        ]

    def fake_best_track_to_records(
        points,
        *,
        initialization_time,
    ):
        calls.append(
            (
                "best_track_to_records",
                points,
                initialization_time,
            )
        )

        return [
            "best-track-record",
        ]

    def fake_verify_tracking_workflow(
        tracking_arg,
        records,
        *,
        observation_name,
    ):
        calls.append(
            (
                "verify",
                tracking_arg,
                records,
                observation_name,
            )
        )

        return verification

    def fake_export_verification_case(
        output_dir,
        *,
        tracking_result,
        verification_result,
    ):
        calls.append(
            (
                "export",
                output_dir,
                tracking_result,
                verification_result,
            )
        )

        return Path(
            output_dir
        )

    monkeypatch.setattr(
        "aiweather.verification.pipeline."
        "open_forecast",
        fake_open_forecast,
    )

    monkeypatch.setattr(
        "aiweather.verification.pipeline."
        "evaluate_forecast_trackers",
        fake_evaluate_forecast_trackers,
    )

    monkeypatch.setattr(
        "aiweather.verification.pipeline."
        "read_ibtracs_csv",
        fake_read_ibtracs_csv,
    )

    monkeypatch.setattr(
        "aiweather.verification.pipeline."
        "best_track_to_records",
        fake_best_track_to_records,
    )

    monkeypatch.setattr(
        "aiweather.verification.pipeline."
        "verify_tracking_workflow",
        fake_verify_tracking_workflow,
    )

    monkeypatch.setattr(
        "aiweather.verification.pipeline."
        "export_verification_case",
        fake_export_verification_case,
    )

    output_dir = (
        tmp_path
        / "verification"
    )

    result = run_tc_verification_pipeline(
        "forecast.zarr",
        ibtracs_path="ibtracs.csv",
        sid="2026204N08267",
        lat_min=5.0,
        lat_max=35.0,
        lon_min=-130.0,
        lon_max=-90.0,
        device="cuda",
        minimum_overlap=3,
        output_dir=output_dir,
    )

    assert isinstance(
        result,
        TCVerificationPipelineResult,
    )

    assert result.tracking is tracking
    assert result.verification is verification
    assert result.output_dir == output_dir

    assert calls[0] == (
        "open_forecast",
        "forecast.zarr",
    )

    assert calls[1][0] == "evaluate"

    evaluate_kwargs = calls[1][2]

    assert evaluate_kwargs == {
        "lat_min": 5.0,
        "lat_max": 35.0,
        "lon_min": -130.0,
        "lon_max": -90.0,
        "device": "cuda",
        "minimum_overlap": 3,
    }

    assert calls[2] == (
        "read_ibtracs",
        "ibtracs.csv",
        "2026204N08267",
    )

    assert calls[3][0] == (
        "best_track_to_records"
    )

    assert calls[3][2] == np.datetime64(
        "2026-07-24T00:00:00"
    )

    assert calls[4][0] == "verify"

    assert calls[4][3] == (
        "ibtracs_2026204N08267"
    )

    assert calls[5][0] == "export"


def test_run_tc_verification_pipeline_without_export(
    monkeypatch,
):
    forecast = SimpleNamespace(
        metadata=SimpleNamespace(
            initialization_time=np.datetime64(
                "2026-07-24T00:00:00"
            )
        )
    )

    tracking = make_tracking_result()
    verification = make_verification_result()

    monkeypatch.setattr(
        "aiweather.verification.pipeline."
        "open_forecast",
        lambda path: forecast,
    )

    monkeypatch.setattr(
        "aiweather.verification.pipeline."
        "evaluate_forecast_trackers",
        lambda *args, **kwargs: tracking,
    )

    monkeypatch.setattr(
        "aiweather.verification.pipeline."
        "read_ibtracs_csv",
        lambda *args, **kwargs: [],
    )

    monkeypatch.setattr(
        "aiweather.verification.pipeline."
        "best_track_to_records",
        lambda *args, **kwargs: [],
    )

    monkeypatch.setattr(
        "aiweather.verification.pipeline."
        "verify_tracking_workflow",
        lambda *args, **kwargs: verification,
    )

    def fail_export(*args, **kwargs):
        raise AssertionError(
            "export should not be called"
        )

    monkeypatch.setattr(
        "aiweather.verification.pipeline."
        "export_verification_case",
        fail_export,
    )

    result = run_tc_verification_pipeline(
        "forecast.zarr",
        ibtracs_path="ibtracs.csv",
        sid="2026204N08267",
        lat_min=5.0,
        lat_max=35.0,
        lon_min=-130.0,
        lon_max=-90.0,
    )

    assert result.output_dir is None


def test_run_tc_verification_pipeline_invalid_overlap():
    with pytest.raises(
        ValueError,
        match="minimum_overlap",
    ):
        run_tc_verification_pipeline(
            "forecast.zarr",
            ibtracs_path="ibtracs.csv",
            sid="2026204N08267",
            lat_min=5.0,
            lat_max=35.0,
            lon_min=-130.0,
            lon_max=-90.0,
            minimum_overlap=0,
        )


def test_run_tc_verification_pipeline_invalid_sid_type():
    with pytest.raises(
        TypeError,
        match="sid",
    ):
        run_tc_verification_pipeline(
            "forecast.zarr",
            ibtracs_path="ibtracs.csv",
            sid=123,
            lat_min=5.0,
            lat_max=35.0,
            lon_min=-130.0,
            lon_max=-90.0,
        )


def test_run_tc_verification_pipeline_empty_sid():
    with pytest.raises(
        ValueError,
        match="sid cannot be empty",
    ):
        run_tc_verification_pipeline(
            "forecast.zarr",
            ibtracs_path="ibtracs.csv",
            sid="   ",
            lat_min=5.0,
            lat_max=35.0,
            lon_min=-130.0,
            lon_max=-90.0,
        )

def test_run_tc_verification_pipeline_generates_plots(
    monkeypatch,
    tmp_path,
):
    forecast = SimpleNamespace(
        metadata=SimpleNamespace(
            initialization_time=np.datetime64(
                "2026-07-24T00:00:00"
            )
        )
    )

    tracking = make_tracking_result()
    verification = make_verification_result()

    monkeypatch.setattr(
        "aiweather.verification.pipeline.open_forecast",
        lambda path: forecast,
    )

    monkeypatch.setattr(
        "aiweather.verification.pipeline."
        "evaluate_forecast_trackers",
        lambda *args, **kwargs: tracking,
    )

    monkeypatch.setattr(
        "aiweather.verification.pipeline."
        "read_ibtracs_csv",
        lambda *args, **kwargs: [],
    )

    monkeypatch.setattr(
        "aiweather.verification.pipeline."
        "best_track_to_records",
        lambda *args, **kwargs: [],
    )

    monkeypatch.setattr(
        "aiweather.verification.pipeline."
        "verify_tracking_workflow",
        lambda *args, **kwargs: verification,
    )

    plot_dir = (
        tmp_path
        / "plots"
    )

    expected_paths = {
        "track_map":
            plot_dir / "track_map.png",
        "track_error":
            plot_dir / "track_error.png",
    }

    calls = []

    def fake_generate_plots(
        *,
        tracking,
        verification,
        plot_output_dir,
        sid,
    ):
        calls.append(
            {
                "tracking": tracking,
                "verification": verification,
                "plot_output_dir":
                    plot_output_dir,
                "sid": sid,
            }
        )

        return expected_paths

    monkeypatch.setattr(
        "aiweather.verification.pipeline."
        "_generate_verification_plots",
        fake_generate_plots,
    )

    result = run_tc_verification_pipeline(
        "forecast.zarr",
        ibtracs_path="ibtracs.csv",
        sid="2026204N08267",
        lat_min=5.0,
        lat_max=35.0,
        lon_min=-130.0,
        lon_max=-90.0,
        generate_plots=True,
        plot_output_dir=plot_dir,
    )

    assert result.plot_paths == (
        expected_paths
    )

    assert len(calls) == 1

    assert calls[0][
        "plot_output_dir"
    ] == plot_dir

    assert calls[0]["sid"] == (
        "2026204N08267"
    )


def test_run_tc_verification_pipeline_default_plot_dir(
    monkeypatch,
    tmp_path,
):
    forecast = SimpleNamespace(
        metadata=SimpleNamespace(
            initialization_time=np.datetime64(
                "2026-07-24T00:00:00"
            )
        )
    )

    tracking = make_tracking_result()
    verification = make_verification_result()

    monkeypatch.setattr(
        "aiweather.verification.pipeline.open_forecast",
        lambda path: forecast,
    )

    monkeypatch.setattr(
        "aiweather.verification.pipeline."
        "evaluate_forecast_trackers",
        lambda *args, **kwargs: tracking,
    )

    monkeypatch.setattr(
        "aiweather.verification.pipeline."
        "read_ibtracs_csv",
        lambda *args, **kwargs: [],
    )

    monkeypatch.setattr(
        "aiweather.verification.pipeline."
        "best_track_to_records",
        lambda *args, **kwargs: [],
    )

    monkeypatch.setattr(
        "aiweather.verification.pipeline."
        "verify_tracking_workflow",
        lambda *args, **kwargs: verification,
    )

    monkeypatch.setattr(
        "aiweather.verification.pipeline."
        "export_verification_case",
        lambda output_dir, **kwargs:
            Path(output_dir),
    )

    captured = {}

    def fake_generate_plots(
        *,
        tracking,
        verification,
        plot_output_dir,
        sid,
    ):
        captured[
            "plot_output_dir"
        ] = plot_output_dir

        return {}

    monkeypatch.setattr(
        "aiweather.verification.pipeline."
        "_generate_verification_plots",
        fake_generate_plots,
    )

    output_dir = (
        tmp_path
        / "case"
    )

    run_tc_verification_pipeline(
        "forecast.zarr",
        ibtracs_path="ibtracs.csv",
        sid="2026204N08267",
        lat_min=5.0,
        lat_max=35.0,
        lon_min=-130.0,
        lon_max=-90.0,
        output_dir=output_dir,
        generate_plots=True,
    )

    assert captured[
        "plot_output_dir"
    ] == (
        output_dir
        / "plots"
    )


def test_run_tc_verification_pipeline_invalid_generate_plots():
    with pytest.raises(
        TypeError,
        match="generate_plots",
    ):
        run_tc_verification_pipeline(
            "forecast.zarr",
            ibtracs_path="ibtracs.csv",
            sid="2026204N08267",
            lat_min=5.0,
            lat_max=35.0,
            lon_min=-130.0,
            lon_max=-90.0,
            generate_plots="yes",
        )