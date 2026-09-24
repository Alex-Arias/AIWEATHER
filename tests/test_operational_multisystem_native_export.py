from dataclasses import dataclass

import pytest

from scripts.export_multisystem_native_tracks import (
    select_regional_native_track,
)


@dataclass
class Genesis:
    track_index: int
    genesis_lead_time_hours: int
    latitude: float
    longitude: float
    pressure: float
    max_wind: float
    qualifying_points: int


def make_result(
    *,
    track_index,
    lead,
    longitude,
    pressure,
    qualifying_points=3,
    records=5,
):
    genesis = Genesis(
        track_index=track_index,
        genesis_lead_time_hours=lead,
        latitude=17.0,
        longitude=longitude,
        pressure=pressure,
        max_wind=20.0,
        qualifying_points=qualifying_points,
    )

    return genesis, list(range(records))


def test_eastern_selector_prefers_earliest_genesis():
    """
    Regression test for Cycle 5 AIFS2: a later eastern detection
    must not overwrite the earlier operational Polo detection.
    """
    results = [
        make_result(
            track_index=0,
            lead=24,
            longitude=252.25,   # -107.75
            pressure=96196.48,
            qualifying_points=23,
            records=18,
        ),
        make_result(
            track_index=12,
            lead=132,
            longitude=254.0,    # -106.00
            pressure=99479.59,
            qualifying_points=8,
            records=19,
        ),
    ]

    genesis, records = select_regional_native_track(
        results,
        classification="eastern",
    )

    assert genesis.track_index == 0
    assert genesis.genesis_lead_time_hours == 24
    assert genesis.pressure == pytest.approx(96196.48)
    assert len(records) == 18


def test_selector_separates_western_and_eastern_systems():
    results = [
        make_result(
            track_index=1,
            lead=24,
            longitude=236.25,   # -123.75
            pressure=98652.16,
        ),
        make_result(
            track_index=2,
            lead=24,
            longitude=252.25,   # -107.75
            pressure=96196.48,
        ),
    ]

    western, _ = select_regional_native_track(
        results,
        classification="western",
    )
    eastern, _ = select_regional_native_track(
        results,
        classification="eastern",
    )

    assert western.track_index == 1
    assert eastern.track_index == 2


def test_same_lead_prefers_lower_pressure():
    results = [
        make_result(
            track_index=3,
            lead=24,
            longitude=252.0,
            pressure=99000.0,
        ),
        make_result(
            track_index=4,
            lead=24,
            longitude=253.0,
            pressure=98000.0,
        ),
    ]

    genesis, _ = select_regional_native_track(
        results,
        classification="eastern",
    )

    assert genesis.track_index == 4


def test_selector_returns_none_when_region_has_no_candidate():
    results = [
        make_result(
            track_index=1,
            lead=24,
            longitude=236.0,
            pressure=98500.0,
        ),
    ]

    selected = select_regional_native_track(
        results,
        classification="eastern",
    )

    assert selected is None


def test_selector_rejects_invalid_classification():
    with pytest.raises(ValueError):
        select_regional_native_track(
            [],
            classification="central",
        )
