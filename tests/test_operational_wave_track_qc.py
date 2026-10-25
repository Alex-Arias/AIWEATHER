"""Regression tests for optional operational wave track QC."""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(
    0,
    str(Path(__file__).resolve().parents[1] / "scripts"),
)

from operational_wave_track_qc import filter_native_track


INIT = "20261024T060000"


@pytest.fixture
def track():
    return pd.DataFrame({
        "lead_time_hours": [144, 150, 156],
        "latitude": [22.75, 23.25, 26.0],
        "longitude": [252.0, 251.0, 248.75],
    })


@pytest.fixture
def qc():
    return pd.DataFrame({
        "model": ["aifs2", "pangu6"],
        "cycle_time": [
            "2026-10-24 06:00:00",
            "2026-10-23 06:00:00",
        ],
        "lead_time_hours": [156, 144],
        "qc_action": [
            "exclude_from_derived_variability",
            "exclude_from_derived_variability",
        ],
    })


def test_no_qc_preserves_original(track):
    result = filter_native_track(track, init=INIT)
    pd.testing.assert_frame_equal(result, track)


def test_applicable_qc_excludes_flagged_lead(track, qc):
    result = filter_native_track(
        track, qc, model="aifs2", init=INIT
    )
    assert result["lead_time_hours"].tolist() == [144, 150]


def test_other_cycle_does_not_exclude(track, qc):
    result = filter_native_track(
        track, qc, model="aifs2", init="20261023T060000"
    )
    assert len(result) == 3


def test_other_model_does_not_exclude(track, qc):
    result = filter_native_track(
        track, qc, model="graphcast", init=INIT
    )
    assert len(result) == 3


def test_missing_qc_column_rejected(track, qc):
    with pytest.raises(ValueError, match="missing columns"):
        filter_native_track(
            track,
            qc.drop(columns=["model"]),
            init=INIT,
        )


def test_duplicate_exclusion_rejected(track, qc):
    duplicated = pd.concat(
        [qc, qc.iloc[[0]]],
        ignore_index=True,
    )
    with pytest.raises(ValueError, match="Duplicate"):
        filter_native_track(
            track, duplicated, init=INIT
        )


def test_noninteger_lead_rejected(track, qc):
    modified = qc.copy()
    modified["lead_time_hours"] = (
        modified["lead_time_hours"].astype(float)
    )
    modified.loc[0, "lead_time_hours"] = 155.5
    with pytest.raises(ValueError, match="noninteger"):
        filter_native_track(
            track, modified, init=INIT
        )


def test_original_dataframe_unchanged(track, qc):
    original = track.copy(deep=True)
    filter_native_track(
        track, qc, model="aifs2", init=INIT
    )
    pd.testing.assert_frame_equal(track, original)
