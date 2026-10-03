"""Tests for scripts/compare_tracker_variability.py."""

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Load standalone script.
# ---------------------------------------------------------------------------

SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "compare_tracker_variability.py"
)

SPEC = spec_from_file_location(
    "compare_tracker_variability",
    SCRIPT,
)

ctv = module_from_spec(SPEC)
SPEC.loader.exec_module(ctv)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def spread_frame(
    times,
    r67,
    r90,
    max_distance,
    spread_valid=None,
):
    """Build a minimal variability product for tracker comparison."""

    data = {
        "valid_time": pd.to_datetime(times),
        "r67_km": r67,
        "r90_km": r90,
        "max_distance_km": max_distance,
    }

    if spread_valid is not None:
        data["spread_valid"] = spread_valid

    return pd.DataFrame(data)


# ---------------------------------------------------------------------------
# Frozen spread-valid semantics
# ---------------------------------------------------------------------------

def test_valid_spread_filters_invalid_rows():
    df = spread_frame(
        [
            "2026-10-02 12:00:00",
            "2026-10-02 18:00:00",
            "2026-10-03 00:00:00",
        ],
        [10.0, 20.0, 30.0],
        [15.0, 30.0, 45.0],
        [20.0, 40.0, 60.0],
        [True, False, True],
    )

    result = ctv.valid_spread(df)

    assert len(result) == 2

    assert result["valid_time"].tolist() == [
        pd.Timestamp("2026-10-02 12:00:00"),
        pd.Timestamp("2026-10-03 00:00:00"),
    ]


def test_valid_spread_without_flag_keeps_all_rows():
    df = spread_frame(
        [
            "2026-10-02 12:00:00",
            "2026-10-02 18:00:00",
        ],
        [10.0, 20.0],
        [15.0, 30.0],
        [20.0, 40.0],
    )

    result = ctv.valid_spread(df)

    assert len(result) == 2


# ---------------------------------------------------------------------------
# Independent tracker coverage
# ---------------------------------------------------------------------------

def test_summarize_preserves_11_vs_10_valid_time_asymmetry():
    times_11 = pd.date_range(
        "2026-10-02 12:00:00",
        periods=11,
        freq="6h",
    )

    wuduan = spread_frame(
        times_11,
        np.arange(11, dtype=float) + 10.0,
        np.arange(11, dtype=float) + 20.0,
        np.arange(11, dtype=float) + 30.0,
        [True] * 11,
    )

    # Vitart is deliberately missing the final valid time.
    vitart = spread_frame(
        times_11[:-1],
        np.arange(10, dtype=float) + 5.0,
        np.arange(10, dtype=float) + 15.0,
        np.arange(10, dtype=float) + 25.0,
        [True] * 10,
    )

    ws = ctv.summarize(wuduan)
    vs = ctv.summarize(vitart)

    assert ws["n_times"] == 11
    assert vs["n_times"] == 10

    # The comparison must not fabricate/interpolate Vitart's missing
    # endpoint merely to make both trackers have equal row counts.
    assert (
        wuduan["valid_time"].max()
        > vitart["valid_time"].max()
    )


def test_summarize_excludes_spread_invalid_rows():
    df = spread_frame(
        [
            "2026-10-02 12:00:00",
            "2026-10-02 18:00:00",
            "2026-10-03 00:00:00",
        ],
        [10.0, 999.0, 30.0],
        [20.0, 999.0, 50.0],
        [30.0, 999.0, 70.0],
        [True, False, True],
    )

    result = ctv.summarize(df)

    assert result["n_times"] == 2
    assert result["mean_r67_km"] == pytest.approx(20.0)
    assert result["mean_r90_km"] == pytest.approx(35.0)
    assert result["mean_max_km"] == pytest.approx(50.0)


def test_summarize_empty_valid_set_returns_nan():
    df = spread_frame(
        ["2026-10-02 12:00:00"],
        [10.0],
        [20.0],
        [30.0],
        [False],
    )

    result = ctv.summarize(df)

    assert result["n_times"] == 0
    assert np.isnan(result["mean_r67_km"])
    assert np.isnan(result["mean_r90_km"])
    assert np.isnan(result["mean_max_km"])


# ---------------------------------------------------------------------------
# Historical -> recent contraction
# ---------------------------------------------------------------------------

def test_contraction_uses_only_intersection_of_valid_times():
    historical = spread_frame(
        [
            "2026-10-02 12:00:00",
            "2026-10-02 18:00:00",
            "2026-10-03 00:00:00",
        ],
        [100.0, 200.0, 300.0],
        [200.0, 400.0, 600.0],
        [250.0, 450.0, 650.0],
        [True, True, True],
    )

    recent = spread_frame(
        [
            "2026-10-02 12:00:00",
            "2026-10-02 18:00:00",
        ],
        [50.0, 100.0],
        [100.0, 200.0],
        [150.0, 250.0],
        [True, True],
    )

    c67, c90, n = ctv.contraction(
        historical,
        recent,
    )

    assert n == 2
    assert c67 == pytest.approx(50.0)
    assert c90 == pytest.approx(50.0)


def test_contraction_respects_spread_valid_before_matching():
    historical = spread_frame(
        [
            "2026-10-02 12:00:00",
            "2026-10-02 18:00:00",
        ],
        [100.0, 200.0],
        [200.0, 400.0],
        [250.0, 450.0],
        [True, False],
    )

    recent = spread_frame(
        [
            "2026-10-02 12:00:00",
            "2026-10-02 18:00:00",
        ],
        [75.0, 10.0],
        [150.0, 10.0],
        [175.0, 20.0],
        [True, True],
    )

    c67, c90, n = ctv.contraction(
        historical,
        recent,
    )

    assert n == 1
    assert c67 == pytest.approx(25.0)
    assert c90 == pytest.approx(25.0)


def test_contraction_no_common_valid_times_returns_nan():
    historical = spread_frame(
        ["2026-10-02 12:00:00"],
        [100.0],
        [200.0],
        [250.0],
        [True],
    )

    recent = spread_frame(
        ["2026-10-03 12:00:00"],
        [50.0],
        [100.0],
        [150.0],
        [True],
    )

    c67, c90, n = ctv.contraction(
        historical,
        recent,
    )

    assert n == 0
    assert np.isnan(c67)
    assert np.isnan(c90)


# ---------------------------------------------------------------------------
# Window selection
# ---------------------------------------------------------------------------

def test_select_window_is_inclusive_and_does_not_interpolate():
    df = spread_frame(
        pd.date_range(
            "2026-10-02 06:00:00",
            periods=5,
            freq="6h",
        ),
        [1.0, 2.0, 3.0, 4.0, 5.0],
        [2.0, 3.0, 4.0, 5.0, 6.0],
        [3.0, 4.0, 5.0, 6.0, 7.0],
        [True] * 5,
    )

    result = ctv.select_window(
        df,
        pd.Timestamp("2026-10-02 12:00:00"),
        pd.Timestamp("2026-10-03 00:00:00"),
    )

    assert result["valid_time"].tolist() == [
        pd.Timestamp("2026-10-02 12:00:00"),
        pd.Timestamp("2026-10-02 18:00:00"),
        pd.Timestamp("2026-10-03 00:00:00"),
    ]


def test_select_window_nonoverlap_returns_empty():
    df = spread_frame(
        ["2026-10-02 12:00:00"],
        [10.0],
        [20.0],
        [30.0],
        [True],
    )

    result = ctv.select_window(
        df,
        pd.Timestamp("2026-10-10 00:00:00"),
        pd.Timestamp("2026-10-11 00:00:00"),
    )

    assert result.empty
