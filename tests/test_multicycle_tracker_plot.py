"""Tests for the multicycle operational tracker comparison plotter."""

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "plot_multicycle_tracker_comparison.py"
)

SPEC = spec_from_file_location(
    "plot_multicycle_tracker_comparison",
    SCRIPT,
)

MODULE = module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_first_four_cycle_linestyles_preserved():
    assert MODULE.cycle_linestyle(0) == "-"
    assert MODULE.cycle_linestyle(1) == "--"
    assert MODULE.cycle_linestyle(2) == "-."
    assert MODULE.cycle_linestyle(3) == ":"


def test_additional_cycle_linestyle_available():
    style = MODULE.cycle_linestyle(4)

    assert style == (0, (5, 1))


def test_cycle_linestyle_wraps_safely():
    nstyles = len(MODULE.CYCLE_LINESTYLES)

    assert MODULE.cycle_linestyle(nstyles) == MODULE.cycle_linestyle(0)
    assert MODULE.cycle_linestyle(nstyles + 1) == MODULE.cycle_linestyle(1)


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


def test_track_path_wuduan():
    path = MODULE.track_path(
        "Odalys",
        "aifs2",
        "wuduan",
        "20260922T120000",
    )

    assert path == Path(
        "results/operational/"
        "odalys_20260922T120000/"
        "aifs2/"
        "aifs2_wuduan_track.csv"
    )


def test_track_path_vitart():
    path = MODULE.track_path(
        "Polo",
        "pangu6",
        "vitart",
        "20260921T120000",
    )

    assert path == Path(
        "results/operational/"
        "polo_20260921T120000/"
        "pangu6/"
        "pangu6_vitart_track.csv"
    )


def test_rachel_plotting_domain():
    assert MODULE.DOMAINS["Rachel"] == (
        -125.0,
        -90.0,
        5.0,
        30.0,
    )


def test_invest_92e_track_path_native():
    path = MODULE.track_path(
        "INVEST 92E",
        "graphcast",
        "native",
        "20261006T120000",
    )

    assert path == Path(
        "results/operational/"
        "invest_92e_20261006T120000/"
        "graphcast/"
        "graphcast_track.csv"
    )


def test_invest_92e_track_path_wuduan():
    path = MODULE.track_path(
        "INVEST 92E",
        "aifs2",
        "wuduan",
        "20261007T120000",
    )

    assert path == Path(
        "results/operational/"
        "invest_92e_20261007T120000/"
        "aifs2/"
        "aifs2_wuduan_track.csv"
    )


def test_invest_92e_plotting_domain():
    assert MODULE.DOMAINS["INVEST 92E"] == (
        -120.0,
        -90.0,
        5.0,
        35.0,
    )


def test_empty_multicycle_plot_raises(tmp_path, monkeypatch):
    import sys

    import matplotlib
    import pytest

    matplotlib.use("Agg")

    monkeypatch.chdir(tmp_path)

    output = tmp_path / "empty_comparison.png"

    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(SCRIPT),
            "--inits",
            "20261006T120000",
            "20261007T120000",
            "--storms",
            "INVEST 92E",
            "--output",
            str(output),
        ],
    )

    with pytest.raises(RuntimeError, match="No tracks loaded"):
        MODULE.main()

    assert not output.exists()
