import numpy as np
import pytest

from aiweather.tracking import (
    Earth2StudioTrack,
    earth2studio_track_to_records,
    select_regional_track,
)


def make_track(
    path_id=0,
    lead_time_hours=(0, 6, 12),
    latitude=(10.0, 11.0, 12.0),
    longitude=(250.0, 249.0, 248.0),
    pressure=(100500.0, 100000.0, 99500.0),
    max_wind=(15.0, 18.0, 20.0),
):
    return Earth2StudioTrack(
        path_id=path_id,
        lead_time_hours=np.asarray(
            lead_time_hours,
            dtype=int,
        ),
        latitude=np.asarray(
            latitude,
            dtype=float,
        ),
        longitude=np.asarray(
            longitude,
            dtype=float,
        ),
        pressure=np.asarray(
            pressure,
            dtype=float,
        ),
        max_wind=np.asarray(
            max_wind,
            dtype=float,
        ),
    )


def test_earth2studio_track_length():
    track = make_track()

    assert len(track) == 3


def test_select_regional_track():
    outside = make_track(
        path_id=0,
        latitude=(40.0, 41.0, 42.0),
        longitude=(100.0, 101.0, 102.0),
    )

    inside = make_track(
        path_id=1,
        latitude=(10.0, 15.0, 20.0),
        longitude=(260.0, 250.0, 240.0),
    )

    selected = select_regional_track(
        [outside, inside],
        lat_min=5.0,
        lat_max=35.0,
        lon_min=225.0,
        lon_max=270.0,
    )

    assert selected is not None
    assert selected.path_id == 1


def test_select_regional_track_empty():
    selected = select_regional_track(
        [],
        lat_min=5.0,
        lat_max=35.0,
        lon_min=225.0,
        lon_max=270.0,
    )

    assert selected is None


def test_track_to_records():
    track = make_track()

    records = earth2studio_track_to_records(
        track,
        initialization_time=np.datetime64(
            "2026-07-24T00:00:00"
        ),
    )

    assert len(records) == 3

    first = records[0]
    second = records[1]
    last = records[-1]

    assert first.lead_time_hours == 0
    assert first.valid_time == np.datetime64(
        "2026-07-24T00:00:00"
    )

    assert first.latitude == pytest.approx(10.0)
    assert first.longitude == pytest.approx(250.0)

    assert first.pressure == pytest.approx(
        100500.0
    )
    assert first.pressure_units == "Pa"

    assert first.max_wind == pytest.approx(
        15.0
    )
    assert first.wind_units == "m/s"

    assert first.distance_km is None
    assert first.translation_speed_kmh is None
    assert first.bearing_degrees is None
    assert first.cumulative_distance_km == pytest.approx(
        0.0
    )

    assert second.distance_km is not None
    assert second.translation_speed_kmh is not None
    assert second.bearing_degrees is not None
    assert second.cumulative_distance_km > 0.0

    assert last.cumulative_distance_km > (
        second.cumulative_distance_km
    )


def test_track_to_records_valid_times():
    track = make_track(
        lead_time_hours=(6, 12, 18),
    )

    records = earth2studio_track_to_records(
        track,
        initialization_time="2026-07-24T00:00:00",
    )

    assert records[0].valid_time == np.datetime64(
        "2026-07-24T06:00:00"
    )

    assert records[-1].valid_time == np.datetime64(
        "2026-07-24T18:00:00"
    )

def test_select_regional_track_with_lead_window():
    early = make_track(
        path_id=0,
        lead_time_hours=(0, 6, 12, 18, 24),
        latitude=(18.0, 18.0, 18.0, 18.0, 18.0),
        longitude=(232.0, 231.0, 230.0, 229.0, 228.0),
        pressure=(
            99000.0,
            99000.0,
            99000.0,
            99000.0,
            99000.0,
        ),
        max_wind=(
            20.0,
            20.0,
            20.0,
            20.0,
            20.0,
        ),
    )

    target = make_track(
        path_id=2,
        lead_time_hours=(96, 108, 114, 120),
        latitude=(16.75, 17.5, 17.75, 18.25),
        longitude=(248.75, 248.0, 247.5, 247.0),
        pressure=(
            98952.0,
            98889.0,
            98913.0,
            98929.0,
        ),
        max_wind=(
            19.37,
            20.28,
            20.10,
            19.71,
        ),
    )

    selected = select_regional_track(
        [early, target],
        lat_min=5.0,
        lat_max=35.0,
        lon_min=225.0,
        lon_max=270.0,
        lead_min_hours=54,
    )

    assert selected is not None
    assert selected.path_id == 2


def test_select_regional_track_invalid_lead_window():
    with pytest.raises(ValueError):
        select_regional_track(
            [],
            lat_min=5.0,
            lat_max=35.0,
            lon_min=225.0,
            lon_max=270.0,
            lead_min_hours=120,
            lead_max_hours=54,
        )