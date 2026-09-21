import pytest

from aiweather.diagnostics import PressureMinimum
from aiweather.tracking import (
    CandidatePoint,
    CandidateTrack,
    associate_candidates,
)


def minimum(
    pressure,
    latitude,
    longitude,
):
    return PressureMinimum(
        value=pressure,
        latitude=latitude,
        longitude=longitude,
        units="Pa",
    )


def test_candidate_point():
    point = CandidatePoint(
        lead_time_hours=24,
        pressure=100574.0,
        latitude=10.5,
        longitude=259.5,
    )

    assert point.lead_time_hours == 24
    assert point.latitude == pytest.approx(10.5)


def test_candidate_track_properties():
    first = CandidatePoint(
        24,
        100574.0,
        10.5,
        259.5,
    )

    second = CandidatePoint(
        30,
        100683.0,
        11.0,
        258.25,
    )

    track = CandidateTrack(
        points=(first, second)
    )

    assert len(track) == 2
    assert track.first == first
    assert track.last == second


def test_associate_single_candidate():
    candidates = [
        (
            24,
            [
                minimum(
                    100574.0,
                    10.5,
                    259.5,
                )
            ],
        ),
        (
            30,
            [
                minimum(
                    100683.0,
                    11.0,
                    258.25,
                )
            ],
        ),
        (
            36,
            [
                minimum(
                    100444.0,
                    11.5,
                    257.25,
                )
            ],
        ),
    ]

    tracks = associate_candidates(
        candidates,
        maximum_displacement_km=500.0,
    )

    assert len(tracks) == 1
    assert len(tracks[0]) == 3


def test_associate_two_distinct_tracks():
    candidates = [
        (
            24,
            [
                minimum(
                    100500.0,
                    10.0,
                    250.0,
                ),
                minimum(
                    100400.0,
                    30.0,
                    240.0,
                ),
            ],
        ),
        (
            30,
            [
                minimum(
                    100450.0,
                    10.5,
                    249.0,
                ),
                minimum(
                    100350.0,
                    30.5,
                    239.0,
                ),
            ],
        ),
    ]

    tracks = associate_candidates(
        candidates,
        maximum_displacement_km=300.0,
    )

    lengths = sorted(
        len(track)
        for track in tracks
    )

    assert lengths == [2, 2]


def test_distant_candidate_starts_new_track():
    candidates = [
        (
            24,
            [
                minimum(
                    100500.0,
                    10.0,
                    250.0,
                )
            ],
        ),
        (
            30,
            [
                minimum(
                    100400.0,
                    30.0,
                    250.0,
                )
            ],
        ),
    ]

    tracks = associate_candidates(
        candidates,
        maximum_displacement_km=300.0,
    )

    assert len(tracks) == 2
    assert all(
        len(track) == 1
        for track in tracks
    )


def test_nearest_candidate_is_selected():
    candidates = [
        (
            24,
            [
                minimum(
                    100500.0,
                    10.0,
                    250.0,
                )
            ],
        ),
        (
            30,
            [
                minimum(
                    100450.0,
                    10.5,
                    250.0,
                ),
                minimum(
                    100300.0,
                    12.0,
                    250.0,
                ),
            ],
        ),
    ]

    tracks = associate_candidates(
        candidates,
        maximum_displacement_km=500.0,
    )

    continued = [
        track
        for track in tracks
        if len(track) == 2
    ]

    assert len(continued) == 1

    assert continued[0].last.latitude == pytest.approx(
        10.5
    )


def test_candidate_cannot_be_assigned_twice():
    candidates = [
        (
            24,
            [
                minimum(
                    100500.0,
                    10.0,
                    250.0,
                ),
                minimum(
                    100600.0,
                    11.0,
                    250.0,
                ),
            ],
        ),
        (
            30,
            [
                minimum(
                    100400.0,
                    10.5,
                    250.0,
                )
            ],
        ),
    ]

    tracks = associate_candidates(
        candidates,
        maximum_displacement_km=500.0,
    )

    assert sum(
        len(track)
        for track in tracks
    ) == 3


def test_empty_candidates():
    assert associate_candidates([]) == []


def test_invalid_displacement():
    with pytest.raises(ValueError):
        associate_candidates(
            [],
            maximum_displacement_km=0.0,
        )


def test_non_increasing_lead_times():
    candidates = [
        (30, []),
        (24, []),
    ]

    with pytest.raises(ValueError):
        associate_candidates(
            candidates
        )


def test_duplicate_lead_times():
    candidates = [
        (24, []),
        (24, []),
    ]

    with pytest.raises(ValueError):
        associate_candidates(
            candidates
        )


def test_stale_track_cannot_capture_current_candidate():
    candidates = [
        (
            72,
            [
                minimum(
                    100679.0,
                    15.75,
                    253.50,
                )
            ],
        ),
        (
            108,
            [
                minimum(
                    99006.0,
                    15.75,
                    255.00,
                )
            ],
        ),
        (
            114,
            [
                minimum(
                    99158.0,
                    15.75,
                    254.00,
                )
            ],
        ),
    ]

    tracks = associate_candidates(
        candidates,
        maximum_displacement_km=500.0,
        maximum_gap_hours=6,
    )

    continued = [
        track
        for track in tracks
        if (
            track.first.lead_time_hours == 108
            and track.last.lead_time_hours == 114
        )
    ]

    assert len(continued) == 1
    assert len(continued[0]) == 2

    stale = [
        track
        for track in tracks
        if track.first.lead_time_hours == 72
    ]

    assert len(stale) == 1
    assert len(stale[0]) == 1


def test_invalid_maximum_gap_hours():
    with pytest.raises(
        ValueError,
        match="maximum_gap_hours",
    ):
        associate_candidates(
            [],
            maximum_gap_hours=0,
        )
