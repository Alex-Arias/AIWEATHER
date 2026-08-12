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