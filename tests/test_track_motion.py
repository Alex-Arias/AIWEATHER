import pytest

from aiweather.tracking import (
    TrackPoint,
    initial_bearing_degrees,
    track_motion,
)


def make_point(
    lead_time_hours,
    latitude,
    longitude,
):
    return TrackPoint(
        lead_time_hours=lead_time_hours,
        latitude=latitude,
        longitude=longitude,
        pressure=100000.0,
    )


# ---------------------------------------------------------
# Bearing
# ---------------------------------------------------------


def test_bearing_north():
    bearing = initial_bearing_degrees(
        10.0,
        250.0,
        11.0,
        250.0,
    )

    assert bearing == pytest.approx(
        0.0,
        abs=1e-10,
    )


def test_bearing_east():
    bearing = initial_bearing_degrees(
        0.0,
        250.0,
        0.0,
        251.0,
    )

    assert bearing == pytest.approx(
        90.0,
        abs=1e-10,
    )


def test_bearing_west():
    bearing = initial_bearing_degrees(
        0.0,
        250.0,
        0.0,
        249.0,
    )

    assert bearing == pytest.approx(
        270.0,
        abs=1e-10,
    )


def test_bearing_wraps_longitude():
    bearing = initial_bearing_degrees(
        0.0,
        359.0,
        0.0,
        1.0,
    )

    assert bearing == pytest.approx(
        90.0,
        abs=1e-10,
    )


# ---------------------------------------------------------
# Track motion
# ---------------------------------------------------------


def test_track_motion_two_points():
    track = [
        make_point(
            0,
            10.0,
            250.0,
        ),
        make_point(
            6,
            11.0,
            250.0,
        ),
    ]

    motion = track_motion(track)

    assert len(motion) == 1

    segment = motion[0]

    assert segment.lead_time_hours == 6

    assert segment.distance_km == pytest.approx(
        111.2,
        rel=0.02,
    )

    assert segment.translation_speed_kmh == pytest.approx(
        111.2 / 6.0,
        rel=0.02,
    )

    assert segment.bearing_degrees == pytest.approx(
        0.0,
        abs=1e-10,
    )

    assert segment.cumulative_distance_km == pytest.approx(
        segment.distance_km
    )


def test_track_motion_cumulative_distance():
    track = [
        make_point(
            0,
            10.0,
            250.0,
        ),
        make_point(
            6,
            11.0,
            250.0,
        ),
        make_point(
            12,
            12.0,
            250.0,
        ),
    ]

    motion = track_motion(track)

    assert len(motion) == 2

    expected = (
        motion[0].distance_km
        + motion[1].distance_km
    )

    assert motion[1].cumulative_distance_km == pytest.approx(
        expected
    )


def test_track_motion_empty_track():
    assert track_motion([]) == []


def test_track_motion_single_point():
    track = [
        make_point(
            0,
            10.0,
            250.0,
        )
    ]

    assert track_motion(track) == []


def test_track_motion_rejects_non_list():
    with pytest.raises(TypeError):
        track_motion(
            (
                make_point(
                    0,
                    10.0,
                    250.0,
                ),
            )
        )


def test_track_motion_rejects_invalid_object():
    with pytest.raises(TypeError):
        track_motion(
            ["not a TrackPoint"]
        )


def test_track_motion_rejects_duplicate_lead_time():
    track = [
        make_point(
            6,
            10.0,
            250.0,
        ),
        make_point(
            6,
            11.0,
            250.0,
        ),
    ]

    with pytest.raises(ValueError):
        track_motion(track)


def test_track_motion_rejects_decreasing_lead_time():
    track = [
        make_point(
            12,
            10.0,
            250.0,
        ),
        make_point(
            6,
            11.0,
            250.0,
        ),
    ]

    with pytest.raises(ValueError):
        track_motion(track)
