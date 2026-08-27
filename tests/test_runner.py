from pathlib import Path

from aiweather.forecast import ForecastRequest
from aiweather.output import OutputManager
from aiweather.runners import create_runner
import sys
import types

class FakeZarrBackend:
    def __init__(self, path):
        self.path = path


def install_fake_earth2studio_io(
    monkeypatch,
):
    module = types.ModuleType(
        "earth2studio.io"
    )

    module.ZarrBackend = FakeZarrBackend

    monkeypatch.setitem(
        sys.modules,
        "earth2studio.io",
        module,
    )

def test_runner_factory():
    """Verify that the runner factory can create a GraphCast runner."""

    request = ForecastRequest(
        model="graphcast",
        datasource="gfs",
        init_time="2026-07-24T00:00:00",
        lead_time=240,
        device="cpu",
    )

    runner = create_runner(
        request.model
    )

    assert runner is not None
    assert runner.MODEL_NAME == "graphcast"


def test_output_manager_canonical_path(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(
        OutputManager,
        "BASE_OUTPUT",
        tmp_path,
    )

    request = ForecastRequest(
        model="graphcast",
        datasource="gfs",
        init_time="2026-07-24T00:00:00",
        lead_time=240,
        device="cpu",
    )

    path = OutputManager.build_output_path(
        request
    )

    expected = (
        tmp_path
        / "graphcast"
        / "20260724T000000"
        / "forecast.zarr"
    )

    assert path == expected
    assert expected.parent.exists()


def test_runner_build_output_uses_canonical_path(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(
        OutputManager,
        "BASE_OUTPUT",
        tmp_path,
    )

    install_fake_earth2studio_io(
        monkeypatch
    )

    request = ForecastRequest(
        model="graphcast",
        datasource="gfs",
        init_time="2026-07-24T00:00:00",
        lead_time=240,
        device="cpu",
    )

    runner = create_runner(
        request.model
    )

    output_path = runner.build_output(
        request
    )

    expected = (
        tmp_path
        / "graphcast"
        / "20260724T000000"
        / "forecast.zarr"
    )

    assert Path(output_path) == expected
    assert Path(request.output_path) == expected
    assert runner.io is not None
    assert Path(runner.io.path) == expected

def test_runner_build_output_preserves_explicit_path(
    tmp_path,
    monkeypatch,
):
    install_fake_earth2studio_io(
        monkeypatch
    )

    explicit = (
        tmp_path
        / "custom"
        / "forecast.zarr"
    )

    explicit.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    request = ForecastRequest(
        model="graphcast",
        datasource="gfs",
        init_time="2026-07-24T00:00:00",
        lead_time=240,
        output_path=str(explicit),
        device="cpu",
    )

    runner = create_runner(
        request.model
    )

    output_path = runner.build_output(
        request
    )

    assert Path(output_path) == explicit
    assert Path(request.output_path) == explicit
    assert runner.io is not None
    assert Path(runner.io.path) == explicit


def test_pangu3_runner_timestep():
    """
    Verify that Pangu3 uses its native 3-hour forecast cadence.
    """

    runner = create_runner(
        "pangu3"
    )

    assert runner.MODEL_NAME == "pangu3"
    assert runner.MODEL_TIMESTEP == 3


def test_pangu6_runner_timestep():
    """
    Verify that Pangu6 retains its native 6-hour forecast cadence.
    """

    runner = create_runner(
        "pangu6"
    )

    assert runner.MODEL_NAME == "pangu6"
    assert runner.MODEL_TIMESTEP == 6


def test_pangu3_12h_request_uses_four_steps(
    monkeypatch,
):
    """
    Verify conversion of a 12-hour Pangu3 forecast
    into four native 3-hour forecast steps.
    """

    captured = {}

    fake_inference = types.ModuleType(
        "aiweather.backends.earth2studio.inference"
    )

    def fake_run_forecast(
        *,
        time,
        nsteps,
        prognostic,
        data,
        io,
    ):
        captured["time"] = time
        captured["nsteps"] = nsteps
        captured["prognostic"] = prognostic
        captured["data"] = data
        captured["io"] = io

    fake_inference.run_forecast = (
        fake_run_forecast
    )

    monkeypatch.setitem(
        sys.modules,
        "aiweather.backends.earth2studio.inference",
        fake_inference,
    )

    monkeypatch.setattr(
        OutputManager,
        "write_forecast_metadata",
        lambda *args, **kwargs: None,
    )

    request = ForecastRequest(
        model="pangu3",
        datasource="gfs",
        init_time="2026-07-24T00:00:00",
        lead_time=12,
        device="cpu",
    )

    runner = create_runner(
        request.model
    )

    runner.model = object()
    runner.data = object()
    runner.io = object()

    runner.run_forecast(
        request
    )

    assert captured["nsteps"] == 4
    assert captured["time"] == request.init_time
    assert captured["prognostic"] is runner.model
    assert captured["data"] is runner.data
    assert captured["io"] is runner.io


def test_pangu3_rejects_non_divisible_lead_time(
    monkeypatch,
):
    """
    Pangu3 must reject forecast horizons that are not
    divisible by its native 3-hour timestep.
    """

    monkeypatch.setattr(
        OutputManager,
        "write_forecast_metadata",
        lambda *args, **kwargs: None,
    )

    request = ForecastRequest(
        model="pangu3",
        datasource="gfs",
        init_time="2026-07-24T00:00:00",
        lead_time=10,
        device="cpu",
    )

    runner = create_runner(
        request.model
    )

    runner.model = object()
    runner.data = object()
    runner.io = object()

    import pytest

    with pytest.raises(
        ValueError,
        match="divisible by 3 hours",
    ):
        runner.run_forecast(
            request
        )


def test_runner_passes_datasource_source(
    monkeypatch,
):
    """
    Verify that an explicit datasource source is passed
    through the PX runner to the datasource loader.
    """

    import aiweather.backends.earth2studio as backend

    captured = {}
    fake_data = object()

    def fake_load_data_source(
        model_name,
        *,
        datasource,
        source=None,
    ):
        captured["model_name"] = model_name
        captured["datasource"] = datasource
        captured["source"] = source
        return fake_data

    monkeypatch.setattr(
        backend,
        "load_data_source",
        fake_load_data_source,
    )

    request = ForecastRequest(
        model="aifs2",
        datasource="ifs",
        datasource_source="azure",
        init_time="2026-06-29T00:00:00",
        lead_time=240,
        device="cpu",
    )

    runner = create_runner(
        request.model
    )

    runner.load_data(
        request
    )

    assert captured == {
        "model_name": "aifs2",
        "datasource": "ifs",
        "source": "azure",
    }

    assert runner.data is fake_data


def test_runner_datasource_source_defaults_to_none(
    monkeypatch,
):
    """
    Verify that omitting datasource_source preserves the
    datasource implementation's default source behavior.
    """

    import aiweather.backends.earth2studio as backend

    captured = {}

    def fake_load_data_source(
        model_name,
        *,
        datasource,
        source=None,
    ):
        captured["model_name"] = model_name
        captured["datasource"] = datasource
        captured["source"] = source
        return object()

    monkeypatch.setattr(
        backend,
        "load_data_source",
        fake_load_data_source,
    )

    request = ForecastRequest(
        model="aifs2",
        datasource="ifs",
        init_time="2026-06-29T00:00:00",
        lead_time=240,
        device="cpu",
    )

    runner = create_runner(
        request.model
    )

    runner.load_data(
        request
    )

    assert captured == {
        "model_name": "aifs2",
        "datasource": "ifs",
        "source": None,
    }
