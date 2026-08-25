from pathlib import Path
from types import SimpleNamespace

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

from datetime import datetime


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

            initialization_time=datetime(
                2026,
                7,
                24,
                0,
                0,
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

    manifest_calls = []

    def fake_export_verification_manifest(
        output_dir,
        **kwargs,
    ):
        manifest_calls.append(
            (
                Path(output_dir),
                kwargs,
            )
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

    monkeypatch.setattr(
        "aiweather.verification.pipeline."
        "export_verification_manifest",
        fake_export_verification_manifest,
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

    assert calls[1] == (
        "read_ibtracs",
        Path("ibtracs.csv"),
        "2026204N08267",
    )

    assert calls[2][0] == (
        "best_track_to_records"
    )

    assert calls[2][2] == datetime(
        2026,
        7,
        24,
        0,
        0,
    )

    assert calls[3][0] == "evaluate"

    evaluate_kwargs = calls[3][2]

    assert evaluate_kwargs == {
        "lat_min": 5.0,
        "lat_max": 35.0,
        "lon_min": -130.0,
        "lon_max": -90.0,
        "device": "cuda",
        "minimum_overlap": 3,
        "maximum_mean_error_km": None,
        "reference_records": [
            "best-track-record",
        ],
    }

    assert calls[4][0] == "verify"

    assert calls[4][3] == (
        "ibtracs_2026204N08267"
    )

    assert calls[5][0] == "export"

    assert len(manifest_calls) == 1

    manifest_output_dir, manifest_kwargs = (
        manifest_calls[0]
    )

    assert manifest_output_dir == output_dir

    assert manifest_kwargs[
        "forecast_path"
    ] == "forecast.zarr"

    assert manifest_kwargs[
        "forecast_metadata"
    ] is forecast.metadata

    assert manifest_kwargs[
        "sid"
    ] == "2026204N08267"

    assert manifest_kwargs[
        "lat_min"
    ] == 5.0

    assert manifest_kwargs[
        "lat_max"
    ] == 35.0

    assert manifest_kwargs[
        "lon_min"
    ] == -130.0

    assert manifest_kwargs[
        "lon_max"
    ] == -90.0

    assert manifest_kwargs[
        "ibtracs_path"
    ] == Path(
        "ibtracs.csv"
    )

    assert manifest_kwargs[
        "ibtracs_basin"
    ] == "EP"

    assert manifest_kwargs[
        "device"
    ] == "cuda"

    assert manifest_kwargs[
        "minimum_overlap"
    ] == 3

    assert manifest_kwargs[
        "generate_plots"
    ] is False

    assert manifest_kwargs[
        "field_lead_times"
    ] is None


def test_run_tc_verification_pipeline_auto_output_dir(
    monkeypatch,
    tmp_path,
):
    forecast = SimpleNamespace(
        metadata=SimpleNamespace(
            initialization_time=datetime(
                2026,
                7,
                24,
                0,
                0,
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
        "ensure_ibtracs_dataset",
        lambda **kwargs: Path(
            "ibtracs.csv"
        ),
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

    resolved_output = (
        tmp_path
        / "verification"
        / (
            "2026204N08267_"
            "20260724T000000"
        )
    )

    monkeypatch.setattr(
        "aiweather.verification.pipeline."
        "OutputManager."
        "build_verification_output_dir",
        lambda **kwargs: resolved_output,
    )

    captured = {}

    def fake_export(
        output_dir,
        *,
        tracking_result,
        verification_result,
    ):
        captured[
            "output_dir"
        ] = output_dir

        return Path(
            output_dir
        )

    monkeypatch.setattr(
        "aiweather.verification.pipeline."
        "export_verification_case",
        fake_export,
    )

    monkeypatch.setattr(
        "aiweather.verification.pipeline."
        "export_verification_manifest",
        lambda *args, **kwargs: None,
    )

    result = run_tc_verification_pipeline(
        "forecast.zarr",
        sid="2026204N08267",
        lat_min=5.0,
        lat_max=35.0,
        lon_min=-130.0,
        lon_max=-90.0,
        output_dir=None,
    )

    assert captured[
        "output_dir"
    ] == resolved_output

    assert result.output_dir == (
        resolved_output
    )

    assert result.plot_paths == {}

#------------------------------------------------------------------------------------------------------

def test_run_tc_verification_pipeline_auto_output_dir_without_plots(
    monkeypatch,
    tmp_path,
):
    forecast = SimpleNamespace(
        metadata=SimpleNamespace(
            initialization_time=datetime(
                2026,
                7,
                24,
                0,
                0,
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

    monkeypatch.setattr(
        "aiweather.verification.pipeline."
        "OutputManager.BASE_RESULTS",
        tmp_path,
    )

    exported_paths = []

    def fake_export(
        output_dir,
        *,
        tracking_result,
        verification_result,
    ):
        exported_paths.append(
            Path(output_dir)
        )

        assert tracking_result is tracking
        assert verification_result is verification

        return Path(
            output_dir
        )

    monkeypatch.setattr(
        "aiweather.verification.pipeline."
        "export_verification_case",
        fake_export,
    )

    monkeypatch.setattr(
        "aiweather.verification.pipeline."
        "export_verification_manifest",
        lambda *args, **kwargs: None,
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

    expected = (
        tmp_path
        / "verification"
        / "2026204N08267_20260724T000000"
    )

    assert exported_paths == [
        expected,
    ]

    assert result.output_dir == expected
    assert result.plot_paths == {}


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
            initialization_time=datetime(
                2026,
                7,
                24,
                0,
                0,
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

    monkeypatch.setattr(
        "aiweather.verification.pipeline."
        "export_verification_manifest",
        lambda *args, **kwargs: None,
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
        output_dir=(tmp_path / "verification"),
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
            initialization_time=datetime(
                2026,
                7,
                24,
                0,
                0,
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

    monkeypatch.setattr(
        "aiweather.verification.pipeline."
        "export_verification_manifest",
        lambda *args, **kwargs: None,
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

def test_run_tc_verification_pipeline_generates_field_sequence(
    monkeypatch,
    tmp_path,
):
    forecast = SimpleNamespace(
        metadata=SimpleNamespace(
            initialization_time=datetime(
                2026,
                7,
                24,
                0,
                0,
            )
        ),
        dataset=object(),
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

    monkeypatch.setattr(
        "aiweather.verification.pipeline."
        "_generate_verification_plots",
        lambda **kwargs: {},
    )

    calls = []

    expected_path = (
        tmp_path
        / "plots"
        / "field_sequence.png"
    )

    def fake_generate_field_sequence_plot(
        *,
        forecast,
        tracking,
        verification,
        field_lead_times,
        plot_output_dir,
    ):
        calls.append(
            {
                "forecast": forecast,
                "tracking": tracking,
                "verification": verification,
                "field_lead_times":
                    field_lead_times,
                "plot_output_dir":
                    plot_output_dir,
            }
        )

        return expected_path

    monkeypatch.setattr(
        "aiweather.verification.pipeline."
        "_generate_field_sequence_plot",
        fake_generate_field_sequence_plot,
    )

    monkeypatch.setattr(
        "aiweather.verification.pipeline."
        "export_verification_manifest",
        lambda *args, **kwargs: None,
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
        output_dir=(tmp_path / "verification"),
        plot_output_dir=(
            tmp_path / "plots"
        ),
        field_lead_times=[
            54,
            78,
            96,
            120,
        ],
    )

    assert len(calls) == 1

    assert calls[0][
        "field_lead_times"
    ] == [
        54,
        78,
        96,
        120,
    ]

    assert result.plot_paths[
        "field_sequence"
    ] == expected_path


def test_run_tc_verification_pipeline_ignores_field_sequence_without_plots(
    monkeypatch,
    tmp_path,
):
    forecast = SimpleNamespace(
        metadata=SimpleNamespace(
            initialization_time=datetime(
                2026,
                7,
                24,
                0,
                0,
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

    def fail_field_sequence(*args, **kwargs):
        raise AssertionError(
            "field sequence should not be generated"
        )

    monkeypatch.setattr(
        "aiweather.verification.pipeline."
        "_generate_field_sequence_plot",
        fail_field_sequence,
    )

    monkeypatch.setattr(
        "aiweather.verification.pipeline."
        "export_verification_manifest",
        lambda *args, **kwargs: None,
    )

    result = run_tc_verification_pipeline(
        "forecast.zarr",
        ibtracs_path="ibtracs.csv",
        sid="2026204N08267",
        lat_min=5.0,
        lat_max=35.0,
        lon_min=-130.0,
        lon_max=-90.0,
        output_dir=(tmp_path / "verification"),
        generate_plots=False,
        field_lead_times=[
            54,
            78,
        ],
    )

    assert result.plot_paths == {}


def test_run_tc_verification_pipeline_invalid_field_lead_times_type():
    with pytest.raises(
        TypeError,
        match="field_lead_times",
    ):
        run_tc_verification_pipeline(
            "forecast.zarr",
            ibtracs_path="ibtracs.csv",
            sid="2026204N08267",
            lat_min=5.0,
            lat_max=35.0,
            lon_min=-130.0,
            lon_max=-90.0,
            field_lead_times=(
                54,
                78,
            ),
        )


def test_run_tc_verification_pipeline_negative_field_lead_time():
    with pytest.raises(
        ValueError,
        match="negative",
    ):
        run_tc_verification_pipeline(
            "forecast.zarr",
            ibtracs_path="ibtracs.csv",
            sid="2026204N08267",
            lat_min=5.0,
            lat_max=35.0,
            lon_min=-130.0,
            lon_max=-90.0,
            field_lead_times=[
                54,
                -6,
            ],
        )

def test_pipeline_explicit_ibtracs_path_bypasses_cache(
    monkeypatch,
    tmp_path,
):
    forecast = SimpleNamespace(
        metadata=SimpleNamespace(
            initialization_time=datetime(
                2026,
                7,
                24,
                0,
                0,
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

    def fail_ensure(*args, **kwargs):
        raise AssertionError(
            "cache manager should not be called"
        )

    monkeypatch.setattr(
        "aiweather.verification.pipeline."
        "ensure_ibtracs_dataset",
        fail_ensure,
    )

    captured = {}

    def fake_read(
        path,
        *,
        sid,
    ):
        captured["path"] = path
        captured["sid"] = sid
        return []

    monkeypatch.setattr(
        "aiweather.verification.pipeline."
        "read_ibtracs_csv",
        fake_read,
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

    monkeypatch.setattr(
        "aiweather.verification.pipeline."
        "export_verification_manifest",
        lambda *args, **kwargs: None,
    )

    run_tc_verification_pipeline(
        "forecast.zarr",
        sid="2026204N08267",
        lat_min=5.0,
        lat_max=35.0,
        lon_min=-130.0,
        lon_max=-90.0,
        ibtracs_path="frozen.csv",
        output_dir=(tmp_path / "verification"),
    )

    assert captured["path"] == Path(
        "frozen.csv"
    )

    assert captured["sid"] == (
        "2026204N08267"
    )


def test_pipeline_auto_ibtracs_resolution(
    monkeypatch,
    tmp_path,
):
    forecast = SimpleNamespace(
        metadata=SimpleNamespace(
            initialization_time=datetime(
                2026,
                7,
                24,
                0,
                0,
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

    resolved = (
        tmp_path
        / "ibtracs.EP.list.v04r01.csv"
    )

    calls = []

    def fake_ensure(
        *,
        basin,
        cache_dir,
        max_age_hours,
        force_update,
    ):
        calls.append(
            {
                "basin": basin,
                "cache_dir": cache_dir,
                "max_age_hours":
                    max_age_hours,
                "force_update":
                    force_update,
            }
        )

        return resolved

    monkeypatch.setattr(
        "aiweather.verification.pipeline."
        "ensure_ibtracs_dataset",
        fake_ensure,
    )

    captured = {}

    def fake_read(
        path,
        *,
        sid,
    ):
        captured["path"] = path
        return []

    monkeypatch.setattr(
        "aiweather.verification.pipeline."
        "read_ibtracs_csv",
        fake_read,
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

    monkeypatch.setattr(
        "aiweather.verification.pipeline."
        "export_verification_manifest",
        lambda *args, **kwargs: None,
    )

    run_tc_verification_pipeline(
        "forecast.zarr",
        sid="2026204N08267",
        lat_min=5.0,
        lat_max=35.0,
        lon_min=-130.0,
        lon_max=-90.0,
        ibtracs_path=None,
        ibtracs_basin="EP",
        ibtracs_cache_dir=tmp_path,
        ibtracs_max_age_hours=72.0,
        ibtracs_force_update=True,
        output_dir=(tmp_path / "verification"),
    )

    assert calls == [
        {
            "basin": "EP",
            "cache_dir": tmp_path,
            "max_age_hours": 72.0,
            "force_update": True,
        }
    ]

    assert captured["path"] == resolved


def test_pipeline_invalid_ibtracs_force_update():
    with pytest.raises(
        TypeError,
        match="ibtracs_force_update",
    ):
        run_tc_verification_pipeline(
            "forecast.zarr",
            sid="2026204N08267",
            lat_min=5.0,
            lat_max=35.0,
            lon_min=-130.0,
            lon_max=-90.0,
            ibtracs_force_update="yes",
        )


def test_pipeline_invalid_ibtracs_max_age():
    with pytest.raises(
        ValueError,
        match="ibtracs_max_age_hours",
    ):
        run_tc_verification_pipeline(
            "forecast.zarr",
            sid="2026204N08267",
            lat_min=5.0,
            lat_max=35.0,
            lon_min=-130.0,
            lon_max=-90.0,
            ibtracs_max_age_hours=0.0,
        )

def test_pipeline_explicit_output_dir_overrides_auto(
    monkeypatch,
    tmp_path,
):
    forecast = SimpleNamespace(
        metadata=SimpleNamespace(
            initialization_time=datetime(
                2026,
                7,
                24,
                0,
                0,
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
        "ensure_ibtracs_dataset",
        lambda **kwargs: Path(
            "ibtracs.csv"
        ),
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

    def fail_auto_output(*args, **kwargs):
        raise AssertionError(
            "automatic output naming "
            "should not be called"
        )

    monkeypatch.setattr(
        "aiweather.verification.pipeline."
        "OutputManager."
        "build_verification_output_dir",
        fail_auto_output,
    )

    explicit_output = (
        tmp_path
        / "custom_case"
    )

    monkeypatch.setattr(
        "aiweather.verification.pipeline."
        "export_verification_case",
        lambda output_dir, **kwargs:
            Path(output_dir),
    )

    monkeypatch.setattr(
        "aiweather.verification.pipeline."
        "export_verification_manifest",
        lambda *args, **kwargs: None,
    )

    result = run_tc_verification_pipeline(
        "forecast.zarr",
        sid="2026204N08267",
        lat_min=5.0,
        lat_max=35.0,
        lon_min=-130.0,
        lon_max=-90.0,
        output_dir=explicit_output,
    )

    assert result.output_dir == (
        explicit_output
    )

def test_pipeline_missing_initialization_time(
    monkeypatch,
):
    forecast = SimpleNamespace(
        metadata=SimpleNamespace(
            initialization_time=None
        )
    )

    monkeypatch.setattr(
        "aiweather.verification.pipeline.open_forecast",
        lambda path: forecast,
    )

    with pytest.raises(
        ValueError,
        match="initialization_time",
    ):
        run_tc_verification_pipeline(
            "forecast.zarr",
            sid="2026204N08267",
            lat_min=5.0,
            lat_max=35.0,
            lon_min=-130.0,
            lon_max=-90.0,
        )