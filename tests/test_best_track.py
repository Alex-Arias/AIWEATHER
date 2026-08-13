import numpy as np
import pytest

from aiweather.verification.best_track import (
    BestTrackPoint,
    best_track_to_records,
)


INITIALIZATION_TIME = np.datetime64(
    "2026-07-24T00:00:00"
)


def test_best_track_to_records():
    points = [
        BestTrackPoint(
            valid_time=np.datetime64(
                "2026-07-24T06:00:00"
            ),
            latitude=11.0,
            longitude=249.0,
            pressure=99500.0,
            max_wind=20.0,
            pressure_units="Pa",
            wind_units="m/s",
        ),
        BestTrackPoint(
            valid_time=np.datetime64(
                "2026-07-24T00:00:00"
            ),
            latitude=10.0,
            longitude=250.0,
            pressure=100000.0,
            max_wind=18.0,
            pressure_units="Pa",
            wind_units="m/s",
        ),
    ]

    records = best_track_to_records(
        points,
        initialization_time=INITIALIZATION_TIME,
    )

    assert len(records) == 2

    assert records[0].lead_time_hours == 0
    assert records[1].lead_time_hours == 6

    assert records[0].latitude == pytest.approx(
        10.0
    )
    assert records[0].longitude == pytest.approx(
        250.0
    )

    assert records[1].pressure == pytest.approx(
        99500.0
    )
    assert records[1].max_wind == pytest.approx(
        20.0
    )


def test_best_track_to_records_sorts_by_time():
    points = [
        BestTrackPoint(
            valid_time=np.datetime64(
                "2026-07-24T12:00:00"
            ),
            latitude=12.0,
            longitude=248.0,
        ),
        BestTrackPoint(
            valid_time=np.datetime64(
                "2026-07-24T00:00:00"
            ),
            latitude=10.0,
            longitude=250.0,
        ),
        BestTrackPoint(
            valid_time=np.datetime64(
                "2026-07-24T06:00:00"
            ),
            latitude=11.0,
            longitude=249.0,
        ),
    ]

    records = best_track_to_records(
        points,
        initialization_time=INITIALIZATION_TIME,
    )

    assert [
        item.lead_time_hours
        for item in records
    ] == [
        0,
        6,
        12,
    ]


def test_best_track_to_records_optional_intensity():
    points = [
        BestTrackPoint(
            valid_time=np.datetime64(
                "2026-07-24T00:00:00"
            ),
            latitude=10.0,
            longitude=250.0,
        ),
    ]

    records = best_track_to_records(
        points,
        initialization_time=INITIALIZATION_TIME,
    )

    assert records[0].pressure is None
    assert records[0].max_wind is None
    assert records[0].pressure_units is None
    assert records[0].wind_units is None


def test_best_track_to_records_negative_lead():
    points = [
        BestTrackPoint(
            valid_time=np.datetime64(
                "2026-07-23T18:00:00"
            ),
            latitude=9.0,
            longitude=251.0,
        ),
    ]

    records = best_track_to_records(
        points,
        initialization_time=INITIALIZATION_TIME,
    )

    assert records[0].lead_time_hours == -6


def test_best_track_to_records_empty():
    records = best_track_to_records(
        [],
        initialization_time=INITIALIZATION_TIME,
    )

    assert records == []


def test_best_track_to_records_invalid_points_type():
    with pytest.raises(
        TypeError,
        match="points must be a list",
    ):
        best_track_to_records(
            "invalid",
            initialization_time=INITIALIZATION_TIME,
        )


def test_best_track_to_records_invalid_point_contents():
    with pytest.raises(
        TypeError,
        match="BestTrackPoint",
    ):
        best_track_to_records(
            [object()],
            initialization_time=INITIALIZATION_TIME,
        )


def test_best_track_to_records_invalid_initialization():
    with pytest.raises(
        ValueError,
        match="NaT",
    ):
        best_track_to_records(
            [],
            initialization_time=np.datetime64(
                "NaT"
            ),
        )


def test_best_track_to_records_invalid_valid_time():
    points = [
        BestTrackPoint(
            valid_time=np.datetime64(
                "NaT"
            ),
            latitude=10.0,
            longitude=250.0,
        ),
    ]

    with pytest.raises(
        ValueError,
        match="valid_time",
    ):
        best_track_to_records(
            points,
            initialization_time=INITIALIZATION_TIME,
        )