"""Tests for the reusable operational tracker comparison plotter."""

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import pandas as pd


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "plot_operational_tracker_comparison.py"
)

SPEC = spec_from_file_location(
    "plot_operational_tracker_comparison",
    SCRIPT,
)

MODULE = module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_track_path_native():
    path = MODULE.track_path(
        "Polo",
        "graphcast",
        "native",
        "20260923T120000",
    )

    assert path == Path(
        "results/operational/"
        "polo_20260923T120000/"
        "graphcast/"
        "graphcast_track.csv"
    )


def test_track_path_external_tracker():
    path = MODULE.track_path(
        "Odalys",
        "pangu3",
        "wuduan",
        "20260923T120000",
    )

    assert path == Path(
        "results/operational/"
        "odalys_20260923T120000/"
        "pangu3/"
        "pangu3_wuduan_track.csv"
    )


def test_load_track_missing_returns_none(
    monkeypatch,
    tmp_path,
):
    monkeypatch.chdir(tmp_path)

    result = MODULE.load_track(
        "Polo",
        "graphcast",
        "native",
        "20260923T120000",
    )

    assert result is None


def test_load_track_normalizes_longitude(
    monkeypatch,
    tmp_path,
):
    monkeypatch.chdir(tmp_path)

    path = (
        tmp_path
        / "results"
        / "operational"
        / "polo_20260923T120000"
        / "graphcast"
        / "graphcast_track.csv"
    )

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    pd.DataFrame(
        {
            "lead_time_hours": [0, 24],
            "latitude": [15.0, 16.0],
            "longitude": [258.0, 257.0],
        }
    ).to_csv(
        path,
        index=False,
    )

    result = MODULE.load_track(
        "Polo",
        "graphcast",
        "native",
        "20260923T120000",
    )

    assert result is not None
    assert result["lead_time_hours"].tolist() == [
        0.0,
        24.0,
    ]
    assert result["latitude"].tolist() == [
        15.0,
        16.0,
    ]
    assert result["longitude_plot"].tolist() == [
        -102.0,
        -103.0,
    ]


def test_load_track_rejects_missing_columns(
    monkeypatch,
    tmp_path,
):
    monkeypatch.chdir(tmp_path)

    path = (
        tmp_path
        / "results"
        / "operational"
        / "polo_20260923T120000"
        / "graphcast"
        / "graphcast_track.csv"
    )

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    pd.DataFrame(
        {
            "lead_time_hours": [0],
            "latitude": [15.0],
        }
    ).to_csv(
        path,
        index=False,
    )

    try:
        MODULE.load_track(
            "Polo",
            "graphcast",
            "native",
            "20260923T120000",
        )
    except ValueError as exc:
        assert "longitude" in str(exc)
    else:
        raise AssertionError(
            "Expected ValueError for missing longitude"
        )

def test_rachel_plotting_domain():
    assert MODULE.DOMAINS["Rachel"] == (
        -125.0,
        -90.0,
        5.0,
        30.0,
    )

