import numpy as np
import pytest

from aiweather.tracking.comparison import (
    align_tracks_by_valid_time,
    compare_tracks_by_valid_time,
)
from aiweather.tracking.records import TrackRecord


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


def test_align_tracks_by_valid_time_ignores_different_leads():
    """
    Consecutive forecast cycles must align using valid time,
    not forecast lead time.
    """
    previous = [
        record(
            48,
            "2026-09-29T12:00:00",
            20.0,
            -113.0,
        ),
        record(
            54,
            "2026-09-29T18:00:00",
            21.0,
            -113.0,
        ),
    ]

    current = [
        record(
            24,
            "2026-09-28T12:00:00",
            19.0,
            -113.0,
        ),
        record(
            48,
            "2026-09-29T12:00:00",
            20.5,
            -113.0,
        ),
        record(
            54,
            "2026-09-29T18:00:00",
            21.5,
            -113.0,
        ),
    ]

    aligned = align_tracks_by_valid_time(
        previous,
        current,
    )

    assert len(aligned) == 2

    assert aligned[0][0].valid_time == np.datetime64(
        "2026-09-29T12:00:00"
    )
    assert aligned[0][1].valid_time == np.datetime64(
        "2026-09-29T12:00:00"
    )


def test_compare_tracks_by_valid_time_computes_error():
    previous = [
        record(
            72,
            "2026-09-30T12:00:00",
            20.0,
            -113.0,
        ),
    ]

    current = [
        record(
            48,
            "2026-09-30T12:00:00",
            21.0,
            -113.0,
        ),
    ]

    comparison = compare_tracks_by_valid_time(
        previous,
        current,
        tracker_a="previous_cycle",
        tracker_b="current_cycle",
    )

    assert comparison.overlap_count == 1

    # One degree latitude is approximately 111 km.
    assert comparison.mean_track_error_km == pytest.approx(
        111.2,
        abs=1.0,
    )

    assert comparison.table.iloc[0][
        "valid_time"
    ] == np.datetime64(
        "2026-09-30T12:00:00"
    )


def test_compare_tracks_by_valid_time_no_overlap():
    previous = [
        record(
            24,
            "2026-09-28T12:00:00",
            20.0,
            -113.0,
        ),
    ]

    current = [
        record(
            24,
            "2026-09-29T12:00:00",
            20.0,
            -113.0,
        ),
    ]

    comparison = compare_tracks_by_valid_time(
        previous,
        current,
    )

    assert comparison.overlap_count == 0
    assert np.isnan(
        comparison.mean_track_error_km
    )
