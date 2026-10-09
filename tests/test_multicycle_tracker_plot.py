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



def test_prepare_translation_speed_utc_alignment():
    """Forecast lead times must map to absolute UTC timestamps."""
    from datetime import datetime

    import pandas as pd

    track = pd.DataFrame({
        "lead_time_hours": [0, 6, 12],
        "latitude": [15.0, 15.0, 15.0],
        "longitude": [-105.0, -104.5, -104.0],
    })

    init_time = datetime(2026, 10, 8, 0)

    result = MODULE.prepare_translation_speed(
        track,
        init_time,
    )

    expected = pd.to_datetime([
        "2026-10-08 00:00:00",
        "2026-10-08 06:00:00",
        "2026-10-08 12:00:00",
    ])

    pd.testing.assert_series_equal(
        result["valid_time"].reset_index(drop=True),
        pd.Series(expected, name="valid_time"),
    )

    assert pd.isna(
        result["translation_speed_kmh"].iloc[0]
    )

    assert (
        result["translation_speed_kmh"].iloc[1] > 0
    )


def test_prepare_translation_speed_irregular_intervals():
    """Translation speed must respect irregular forecast intervals."""
    from datetime import datetime

    import pandas as pd
    import pytest

    track = pd.DataFrame({
        "lead_time_hours": [0, 3, 9],
        "latitude": [0.0, 0.0, 0.0],
        "longitude": [0.0, 1.0, 2.0],
    })

    result = MODULE.prepare_translation_speed(
        track,
        datetime(2026, 10, 8, 0),
    )

    assert result["translation_speed_kmh"].iloc[1] == pytest.approx(
        37.065,
        rel=0.001,
    )

    assert result["translation_speed_kmh"].iloc[2] == pytest.approx(
        18.5325,
        rel=0.001,
    )


def test_prepare_translation_speed_multicycle_alignment():
    """Different cycles must share the same absolute UTC reference."""
    from datetime import datetime

    import pandas as pd
    import pytest

    cycle1 = pd.DataFrame({
        "lead_time_hours": [0, 24, 48],
        "latitude": [15.0, 16.0, 17.0],
        "longitude": [-105.0, -106.0, -107.0],
    })

    cycle2 = pd.DataFrame({
        "lead_time_hours": [0, 12, 24],
        "latitude": [16.0, 16.5, 17.0],
        "longitude": [-106.0, -106.5, -107.0],
    })

    result1 = MODULE.prepare_translation_speed(
        cycle1,
        datetime(2026, 10, 6, 12),
    )

    result2 = MODULE.prepare_translation_speed(
        cycle2,
        datetime(2026, 10, 7, 12),
    )

    # Cycle 1 +48 h and Cycle 2 +24 h
    # correspond to the same valid UTC time.
    assert (
        result1["valid_time"].iloc[-1]
        == result2["valid_time"].iloc[-1]
    )

    assert (
        result1["valid_time"].iloc[-1]
        == pd.Timestamp("2026-10-08 12:00:00")
    )

    # Their translation speeds are computed independently.
    assert result1["translation_speed_kmh"].iloc[-1] > 0
    assert result2["translation_speed_kmh"].iloc[-1] > 0
