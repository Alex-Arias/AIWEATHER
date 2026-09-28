from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

from aiweather.tracking.records import TrackRecord

from scripts.export_multisystem_native_tracks import (
    remove_native_track_products,
    select_continuous_native_track,
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



def record(
    lead,
    valid_time,
    lat,
    lon,
):
    return TrackRecord(
        lead_time_hours=lead,
        valid_time=np.datetime64(valid_time),
        latitude=lat,
        longitude=lon,
        pressure=None,
        pressure_units=None,
        max_wind=None,
        wind_units=None,
        distance_km=None,
        translation_speed_kmh=None,
        bearing_degrees=None,
        cumulative_distance_km=0.0,
    )


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



def test_continuity_selector_prefers_matching_polo_candidate():
    """
    Cycle-8 regression.

    Candidate 0 continues the previous-cycle Polo track, whereas
    candidate 1 is a separate eastern-Pacific developing system.
    """
    previous = [
        record(
            lead=24,
            valid_time="2026-09-28T12:00",
            lat=23.5,
            lon=246.5,
        ),
        record(
            lead=30,
            valid_time="2026-09-28T18:00",
            lat=24.0,
            lon=246.0,
        ),
        record(
            lead=36,
            valid_time="2026-09-29T00:00",
            lat=24.5,
            lon=245.5,
        ),
    ]

    polo_records = [
        record(
            lead=24,
            valid_time="2026-09-28T12:00",
            lat=23.6,
            lon=246.4,
        ),
        record(
            lead=30,
            valid_time="2026-09-28T18:00",
            lat=24.1,
            lon=245.9,
        ),
        record(
            lead=36,
            valid_time="2026-09-29T00:00",
            lat=24.6,
            lon=245.4,
        ),
    ]

    competing_records = [
        record(
            lead=24,
            valid_time="2026-09-28T12:00",
            lat=15.5,
            lon=254.0,
        ),
        record(
            lead=30,
            valid_time="2026-09-28T18:00",
            lat=16.0,
            lon=253.5,
        ),
        record(
            lead=36,
            valid_time="2026-09-29T00:00",
            lat=16.5,
            lon=253.0,
        ),
    ]

    results = [
        (
            make_result(
                track_index=0,
                lead=24,
                longitude=246.75,
                pressure=97000.0,
            )[0],
            polo_records,
        ),
        (
            make_result(
                track_index=1,
                lead=24,
                longitude=254.0,
                pressure=99000.0,
            )[0],
            competing_records,
        ),
    ]

    selected = select_continuous_native_track(
        results,
        reference_records=previous,
        minimum_overlap=2,
        maximum_mean_error_km=600.0,
    )

    assert selected is not None

    genesis, records, diagnostics = selected

    assert genesis.track_index == 0
    assert records is polo_records
    assert diagnostics["overlap_count"] == 3
    assert diagnostics["mean_track_error_km"] < 600.0


def test_continuity_selector_rejects_distant_system():
    """
    A measurable candidate exceeding the continuity threshold must
    not inherit the previous storm identity.
    """
    previous = [
        record(
            lead=24,
            valid_time="2026-09-28T12:00",
            lat=24.0,
            lon=246.0,
        ),
        record(
            lead=30,
            valid_time="2026-09-28T18:00",
            lat=24.5,
            lon=245.5,
        ),
    ]

    distant = [
        record(
            lead=24,
            valid_time="2026-09-28T12:00",
            lat=15.0,
            lon=254.0,
        ),
        record(
            lead=30,
            valid_time="2026-09-28T18:00",
            lat=15.5,
            lon=253.5,
        ),
    ]

    genesis = make_result(
        track_index=1,
        lead=24,
        longitude=254.0,
        pressure=99000.0,
    )[0]

    selected = select_continuous_native_track(
        [(genesis, distant)],
        reference_records=previous,
        minimum_overlap=2,
        maximum_mean_error_km=600.0,
    )

    assert selected is None


def test_continuity_selector_rejects_insufficient_overlap():
    previous = [
        record(
            lead=24,
            valid_time="2026-09-28T12:00",
            lat=24.0,
            lon=246.0,
        ),
        record(
            lead=30,
            valid_time="2026-09-28T18:00",
            lat=24.5,
            lon=245.5,
        ),
    ]

    candidate = [
        record(
            lead=24,
            valid_time="2026-09-28T12:00",
            lat=24.1,
            lon=246.1,
        ),
    ]

    genesis = make_result(
        track_index=0,
        lead=24,
        longitude=246.1,
        pressure=97000.0,
    )[0]

    selected = select_continuous_native_track(
        [(genesis, candidate)],
        reference_records=previous,
        minimum_overlap=2,
        maximum_mean_error_km=600.0,
    )

    assert selected is None



def test_remove_native_track_products_removes_only_native(
    tmp_path,
    monkeypatch,
):
    monkeypatch.chdir(tmp_path)

    output_dir = (
        Path("results")
        / "operational"
        / "polo_20260927T120000"
        / "pangu3"
    )
    output_dir.mkdir(parents=True)

    native_csv = output_dir / "pangu3_track.csv"
    native_provenance = (
        output_dir / "pangu3_track.provenance.json"
    )
    wuduan_csv = output_dir / "pangu3_wuduan_track.csv"
    wuduan_provenance = (
        output_dir / "pangu3_wuduan_track.provenance.json"
    )
    vitart_csv = output_dir / "pangu3_vitart_track.csv"
    vitart_provenance = (
        output_dir / "pangu3_vitart_track.provenance.json"
    )

    for product in (
        native_csv,
        native_provenance,
        wuduan_csv,
        wuduan_provenance,
        vitart_csv,
        vitart_provenance,
    ):
        product.write_text("test\n")

    removed = remove_native_track_products(
        storm_name="Polo",
        init="20260927T120000",
        model="pangu3",
    )

    assert set(removed) == {
        native_csv,
        native_provenance,
    }

    assert not native_csv.exists()
    assert not native_provenance.exists()

    assert wuduan_csv.exists()
    assert wuduan_provenance.exists()
    assert vitart_csv.exists()
    assert vitart_provenance.exists()


def test_remove_native_track_products_is_safe_when_absent(
    tmp_path,
    monkeypatch,
):
    monkeypatch.chdir(tmp_path)

    removed = remove_native_track_products(
        storm_name="Polo",
        init="20260927T120000",
        model="pangu6",
    )

    assert removed == []
