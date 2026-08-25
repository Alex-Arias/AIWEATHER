import numpy as np
import pytest

from aiweather.tracking.earth2studio import (
    Earth2StudioTrack,
)
from aiweather.tracking.records import TrackRecord
from aiweather.tracking.selection import (
    TrackMatch,
    select_matching_track,
)


INITIALIZATION_TIME = np.datetime64(
    "2026-07-24T00:00:00"
)


def make_record(
    lead_time_hours,
    latitude,
    longitude,
):
    return TrackRecord(
        lead_time_hours=lead_time_hours,
        valid_time=(
            INITIALIZATION_TIME
            + np.timedelta64(
                lead_time_hours,
                "h",
            )
        ),
        latitude=latitude,
        longitude=longitude,
        pressure=100000.0,
        pressure_units="Pa",
        max_wind=20.0,
        wind_units="m/s",
    )


def make_earth2studio_track(
    path_id,
    lead_times,
    latitudes,
    longitudes,
):
    count = len(lead_times)

    return Earth2StudioTrack(
        path_id=path_id,
        lead_time_hours=np.asarray(
            lead_times,
            dtype=int,
        ),
        latitude=np.asarray(
            latitudes,
            dtype=float,
        ),
        longitude=np.asarray(
            longitudes,
            dtype=float,
        ),
        pressure=np.full(
            count,
            100000.0,
            dtype=float,
        ),
        max_wind=np.full(
            count,
            20.0,
            dtype=float,
        ),
    )


def test_select_matching_track_exact_match():
    reference = [
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

    wrong = make_earth2studio_track(
        0,
        [0, 6, 12],
        [30.0, 31.0, 32.0],
        [200.0, 199.0, 198.0],
    )

    correct = make_earth2studio_track(
        2,
        [0, 6, 12],
        [10.0, 11.0, 12.0],
        [250.0, 249.0, 248.0],
    )

    match = select_matching_track(
        reference,
        [
            wrong,
            correct,
        ],
        initialization_time=(
            INITIALIZATION_TIME
        ),
    )

    assert isinstance(
        match,
        TrackMatch,
    )

    assert match.path_id == 2
    assert match.overlap_count == 3

    assert (
        match.mean_track_error_km
        == pytest.approx(0.0)
    )


def test_select_matching_track_uses_overlap():
    reference = [
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

    candidate = make_earth2studio_track(
        5,
        [6, 12, 18],
        [11.0, 12.0, 13.0],
        [249.0, 248.0, 247.0],
    )

    match = select_matching_track(
        reference,
        [candidate],
        initialization_time=(
            INITIALIZATION_TIME
        ),
    )

    assert match is not None
    assert match.path_id == 5
    assert match.overlap_count == 2

    assert (
        match.mean_track_error_km
        == pytest.approx(0.0)
    )


def test_select_matching_track_minimum_overlap():
    reference = [
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

    candidate = make_earth2studio_track(
        1,
        [12, 18],
        [12.0, 13.0],
        [248.0, 247.0],
    )

    match = select_matching_track(
        reference,
        [candidate],
        initialization_time=(
            INITIALIZATION_TIME
        ),
        minimum_overlap=2,
    )

    assert match is None


def test_select_matching_track_no_candidates():
    reference = [
        make_record(
            0,
            10.0,
            250.0,
        ),
    ]

    match = select_matching_track(
        reference,
        [],
        initialization_time=(
            INITIALIZATION_TIME
        ),
    )

    assert match is None


def test_select_matching_track_prefers_lower_error():
    reference = [
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
    ]

    farther = make_earth2studio_track(
        1,
        [0, 6],
        [11.0, 12.0],
        [250.0, 249.0],
    )

    closer = make_earth2studio_track(
        7,
        [0, 6],
        [10.1, 11.1],
        [250.0, 249.0],
    )

    match = select_matching_track(
        reference,
        [
            farther,
            closer,
        ],
        initialization_time=(
            INITIALIZATION_TIME
        ),
    )

    assert match is not None
    assert match.path_id == 7


def test_select_matching_track_returns_records():
    reference = [
        make_record(
            0,
            10.0,
            250.0,
        ),
    ]

    candidate = make_earth2studio_track(
        3,
        [0],
        [10.0],
        [250.0],
    )

    match = select_matching_track(
        reference,
        [candidate],
        initialization_time=(
            INITIALIZATION_TIME
        ),
    )

    assert match is not None
    assert len(match.records) == 1

    assert (
        match.records[0].lead_time_hours
        == 0
    )


def test_select_matching_track_rejects_invalid_overlap():
    with pytest.raises(
        ValueError,
        match="minimum_overlap",
    ):
        select_matching_track(
            [],
            [],
            initialization_time=(
                INITIALIZATION_TIME
            ),
            minimum_overlap=0,
        )



def test_select_matching_track_rejects_large_mean_error():
    reference = [
        make_record(
            48,
            18.0,
            228.0,
        ),
        make_record(
            54,
            18.2,
            227.5,
        ),
        make_record(
            60,
            18.4,
            227.0,
        ),
    ]

    distant = make_earth2studio_track(
        path_id=0,
        lead_times=[
            48,
            54,
            60,
        ],
        latitudes=[
            35.0,
            35.5,
            36.0,
        ],
        longitudes=[
            150.0,
            149.5,
            149.0,
        ],
    )

    result = select_matching_track(
        reference,
        [distant],
        initialization_time=INITIALIZATION_TIME,
        minimum_overlap=3,
        maximum_mean_error_km=500.0,
    )

    assert result is None
