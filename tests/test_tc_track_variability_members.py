"""Tests for scripts/tc_track_variability_members.py."""

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Load standalone script.
# ---------------------------------------------------------------------------

SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "tc_track_variability_members.py"
)

SPEC = spec_from_file_location(
    "tc_track_variability_members",
    SCRIPT,
)

tcvm = module_from_spec(SPEC)
SPEC.loader.exec_module(tcvm)


# ---------------------------------------------------------------------------
# Cycle discovery
# ---------------------------------------------------------------------------

def test_discover_cycles_orders_chronologically(tmp_path):

    for name in [
        "odalys_20260922T120000",
        "odalys_20260920T120000",
        "odalys_20260921T120000",
    ]:
        (tmp_path / name).mkdir()

    cycles = tcvm.discover_cycles(
        tmp_path,
        "Odalys",
    )

    times = [
        cycle_time
        for cycle_time, _ in cycles
    ]

    assert times == [
        pd.Timestamp("2026-09-20 12:00:00"),
        pd.Timestamp("2026-09-21 12:00:00"),
        pd.Timestamp("2026-09-22 12:00:00"),
    ]


def test_discover_cycles_ignores_other_storms(tmp_path):

    (tmp_path / "odalys_20260920T120000").mkdir()
    (tmp_path / "polo_20260920T120000").mkdir()

    cycles = tcvm.discover_cycles(
        tmp_path,
        "Odalys",
    )

    assert len(cycles) == 1

    assert (
        cycles[0][1].name
        == "odalys_20260920T120000"
    )


def test_discover_cycles_ignores_invalid_cycle_names(tmp_path):

    (tmp_path / "odalys_20260920T120000").mkdir()
    (tmp_path / "odalys_multicycle").mkdir()
    (tmp_path / "odalys_not_a_date").mkdir()

    cycles = tcvm.discover_cycles(
        tmp_path,
        "Odalys",
    )

    assert len(cycles) == 1

    assert cycles[0][0] == pd.Timestamp(
        "2026-09-20 12:00:00"
    )


# ---------------------------------------------------------------------------
# Track loading
# ---------------------------------------------------------------------------

def write_track(path, rows):

    pd.DataFrame(rows).to_csv(
        path,
        index=False,
    )


def test_load_track_keeps_synoptic_six_hour_times(tmp_path):

    track = tmp_path / "track.csv"

    write_track(
        track,
        [
            {
                "lead_time_hours": 0,
                "valid_time": "2026-09-20 12:00:00",
                "latitude": 15.0,
                "longitude": 230.0,
            },
            {
                "lead_time_hours": 3,
                "valid_time": "2026-09-20 15:00:00",
                "latitude": 15.2,
                "longitude": 230.2,
            },
            {
                "lead_time_hours": 6,
                "valid_time": "2026-09-20 18:00:00",
                "latitude": 15.4,
                "longitude": 230.4,
            },
            {
                "lead_time_hours": 12,
                "valid_time": "2026-09-21 00:00:00",
                "latitude": 15.8,
                "longitude": 230.8,
            },
        ],
    )

    result = tcvm.load_track(
        track,
        cycle_number=1,
        cycle_time=pd.Timestamp(
            "2026-09-20 12:00:00"
        ),
        model="graphcast",
    )

    assert len(result) == 3

    assert result[
        "valid_time"
    ].dt.hour.tolist() == [
        12,
        18,
        0,
    ]

    assert result[
        "lead_time_hours"
    ].tolist() == [
        0,
        6,
        12,
    ]


def test_load_track_adds_member_metadata(tmp_path):

    track = tmp_path / "track.csv"

    write_track(
        track,
        [
            {
                "lead_time_hours": 0,
                "valid_time": "2026-09-20 12:00:00",
                "latitude": 15.0,
                "longitude": -110.0,
            },
        ],
    )

    cycle_time = pd.Timestamp(
        "2026-09-20 12:00:00"
    )

    result = tcvm.load_track(
        track,
        cycle_number=4,
        cycle_time=cycle_time,
        model="aifs2",
    )

    assert result["cycle"].tolist() == [4]
    assert result["model"].tolist() == ["aifs2"]

    assert result[
        "cycle_time"
    ].tolist() == [
        cycle_time
    ]


def test_load_track_drops_missing_positions(tmp_path):

    track = tmp_path / "track.csv"

    write_track(
        track,
        [
            {
                "lead_time_hours": 0,
                "valid_time": "2026-09-20 12:00:00",
                "latitude": 15.0,
                "longitude": -110.0,
            },
            {
                "lead_time_hours": 6,
                "valid_time": "2026-09-20 18:00:00",
                "latitude": None,
                "longitude": -110.5,
            },
            {
                "lead_time_hours": 12,
                "valid_time": "2026-09-21 00:00:00",
                "latitude": 16.0,
                "longitude": None,
            },
        ],
    )

    result = tcvm.load_track(
        track,
        cycle_number=1,
        cycle_time=pd.Timestamp(
            "2026-09-20 12:00:00"
        ),
        model="pangu3",
    )

    assert len(result) == 1

    assert result.iloc[0]["latitude"] == pytest.approx(
        15.0
    )


def test_load_track_rejects_missing_required_columns(tmp_path):

    track = tmp_path / "track.csv"

    pd.DataFrame(
        {
            "valid_time": [
                "2026-09-20 12:00:00",
            ],
            "latitude": [15.0],
            "longitude": [-110.0],
        }
    ).to_csv(
        track,
        index=False,
    )

    with pytest.raises(
        ValueError,
        match="missing required columns",
    ):
        tcvm.load_track(
            track,
            cycle_number=1,
            cycle_time=pd.Timestamp(
                "2026-09-20 12:00:00"
            ),
            model="graphcast",
        )


def test_load_track_preserves_longitude_convention(tmp_path):

    track = tmp_path / "track.csv"

    write_track(
        track,
        [
            {
                "lead_time_hours": 0,
                "valid_time": "2026-09-20 12:00:00",
                "latitude": 15.0,
                "longitude": 240.0,
            },
        ],
    )

    result = tcvm.load_track(
        track,
        cycle_number=1,
        cycle_time=pd.Timestamp(
            "2026-09-20 12:00:00"
        ),
        model="graphcast",
    )

    # The member builder preserves source coordinates.
    # Longitude normalization belongs to the downstream
    # variability calculation.
    assert result.iloc[0]["longitude"] == pytest.approx(
        240.0
    )


def test_storm_slug_with_space():
    from scripts.tc_track_variability_members import storm_slug

    assert storm_slug("INVEST 92E") == "invest_92e"
    assert storm_slug("invest_92e") == "invest_92e"


def test_discover_cycles_with_spaced_storm_name(tmp_path):
    from scripts.tc_track_variability_members import discover_cycles

    (tmp_path / "invest_92e_20261006T120000").mkdir()
    (tmp_path / "invest_92e_20261007T120000").mkdir()

    # Unrelated operational case must not be selected.
    (tmp_path / "rachel_20261005T120000").mkdir()

    cycles = discover_cycles(
        tmp_path,
        "INVEST 92E",
    )

    assert len(cycles) == 2

    directories = [
        item[1].name
        for item in cycles
    ]

    assert directories == [
        "invest_92e_20261006T120000",
        "invest_92e_20261007T120000",
    ]
