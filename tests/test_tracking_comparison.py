import numpy as np
import pytest

from aiweather.tracking.comparison import (
    TrackComparison,
    align_tracks,
    compare_tracks,
)
from aiweather.tracking.records import TrackRecord


def make_record(
    lead_time_hours,
    latitude,
    longitude,
    pressure=100000.0,
    max_wind=20.0,
):
    return TrackRecord(
        lead_time_hours=lead_time_hours,
        valid_time=(
            np.datetime64("2026-07-24T00:00:00")
            + np.timedelta64(
                lead_time_hours,
                "h",
            )
        ),
        latitude=latitude,
        longitude=longitude,
        pressure=pressure,
        pressure_units="Pa",
        max_wind=max_wind,
        wind_units="m/s",
    )


def test_align_tracks_common_leads():
    track_a = [
        make_record(0, 10.0, 250.0),
        make_record(6, 11.0, 249.0),
        make_record(12, 12.0, 248.0),
    ]

    track_b = [
        make_record(6, 11.1, 249.1),
        make_record(12, 12.1, 248.1),
        make_record(18, 13.0, 247.0),
    ]

    aligned = align_tracks(
        track_a,
        track_b,
    )

    assert len(aligned) == 2

    assert (
        aligned[0][0].lead_time_hours
        == 6
    )
    assert (
        aligned[0][1].lead_time_hours
        == 6
    )

    assert (
        aligned[1][0].lead_time_hours
        == 12
    )
    assert (
        aligned[1][1].lead_time_hours
        == 12
    )


def test_align_tracks_no_overlap():
    track_a = [
        make_record(
            0,
            10.0,
            250.0,
        ),
    ]

    track_b = [
        make_record(
            6,
            11.0,
            249.0,
        ),
    ]

    aligned = align_tracks(
        track_a,
        track_b,
    )

    assert aligned == []


def test_compare_identical_tracks():
    track = [
        make_record(
            0,
            10.0,
            250.0,
            pressure=100000.0,
            max_wind=18.0,
        ),
        make_record(
            6,
            11.0,
            249.0,
            pressure=99500.0,
            max_wind=20.0,
        ),
    ]

    comparison = compare_tracks(
        track,
        track,
        tracker_a="native",
        tracker_b="native_copy",
    )

    assert isinstance(
        comparison,
        TrackComparison,
    )

    assert comparison.overlap_count == 2

    assert (
        comparison.mean_track_error_km
        == pytest.approx(0.0)
    )

    assert (
        comparison.rmse_track_error_km
        == pytest.approx(0.0)
    )

    assert (
        comparison.median_track_error_km
        == pytest.approx(0.0)
    )

    assert (
        comparison.maximum_track_error_km
        == pytest.approx(0.0)
    )

    assert np.allclose(
        comparison.table[
            "pressure_difference"
        ],
        0.0,
    )

    assert np.allclose(
        comparison.table[
            "wind_difference"
        ],
        0.0,
    )


def test_compare_tracks_known_offset():
    track_a = [
        make_record(
            0,
            0.0,
            0.0,
            pressure=100000.0,
            max_wind=20.0,
        ),
    ]

    track_b = [
        make_record(
            0,
            0.0,
            1.0,
            pressure=99900.0,
            max_wind=18.0,
        ),
    ]

    comparison = compare_tracks(
        track_a,
        track_b,
        tracker_a="a",
        tracker_b="b",
    )

    assert comparison.overlap_count == 1

    # One degree of longitude at the equator is
    # approximately 111 km.
    assert (
        comparison.mean_track_error_km
        == pytest.approx(
            111.2,
            abs=0.5,
        )
    )

    row = comparison.table.iloc[0]

    assert (
        row["pressure_difference"]
        == pytest.approx(100.0)
    )

    assert (
        row["wind_difference"]
        == pytest.approx(2.0)
    )


def test_compare_tracks_partial_overlap():
    track_a = [
        make_record(
            0,
            10.0,
            250.0,
        ),
        make_record(
            6,
            11.0,
            249.0,
        ),
        make_record(
            12,
            12.0,
            248.0,
        ),
    ]

    track_b = [
        make_record(
            6,
            11.0,
            249.0,
        ),
        make_record(
            12,
            12.0,
            248.0,
        ),
    ]

    comparison = compare_tracks(
        track_a,
        track_b,
    )

    assert comparison.overlap_count == 2

    assert list(
        comparison.table[
            "lead_time_hours"
        ]
    ) == [
        6,
        12,
    ]


def test_compare_tracks_empty_overlap():
    comparison = compare_tracks(
        [
            make_record(
                0,
                10.0,
                250.0,
            ),
        ],
        [
            make_record(
                6,
                11.0,
                249.0,
            ),
        ],
    )

    assert comparison.overlap_count == 0
    assert comparison.table.empty

    assert np.isnan(
        comparison.mean_track_error_km
    )

    assert np.isnan(
        comparison.rmse_track_error_km
    )

    assert np.isnan(
        comparison.median_track_error_km
    )

    assert np.isnan(
        comparison.maximum_track_error_km
    )


def test_compare_tracks_missing_intensity():
    track_a = [
        make_record(
            0,
            10.0,
            250.0,
            max_wind=None,
        ),
    ]

    track_b = [
        make_record(
            0,
            10.0,
            250.0,
            max_wind=20.0,
        ),
    ]

    comparison = compare_tracks(
        track_a,
        track_b,
    )

    row = comparison.table.iloc[0]

    assert np.isnan(
        row["wind_difference"]
    )