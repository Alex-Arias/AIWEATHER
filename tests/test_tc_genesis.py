import numpy as np
import pytest
import xarray as xr

from aiweather.tracking import (
    CandidatePoint,
    CandidateTrack,
    GenesisResult,
    detect_genesis,
    select_first_genesis,
)


def make_track(
    *,
    pressures,
    latitudes,
    longitudes,
    lead_times,
):
    points = tuple(
        CandidatePoint(
            lead_time_hours=lead,
            pressure=pressure,
            latitude=latitude,
            longitude=longitude,
        )
        for (
            lead,
            pressure,
            latitude,
            longitude,
        ) in zip(
            lead_times,
            pressures,
            latitudes,
            longitudes,
        )
    )

    return CandidateTrack(
        points=points
    )


def make_wind(
    lead_times,
    values,
):
    u = xr.DataArray(
        np.asarray(values, dtype=float).reshape(
            len(lead_times),
            1,
            1,
        ),
        dims=(
            "lead_time",
            "lat",
            "lon",
        ),
        coords={
            "lead_time": [
                np.timedelta64(
                    lead,
                    "h",
                )
                for lead in lead_times
            ],
            "lat": [10.0],
            "lon": [250.0],
        },
    )

    v = xr.zeros_like(
        u
    )

    return u, v


def test_detect_genesis_known_track():
    track = make_track(
        pressures=[
            100800.0,
            100400.0,
            100300.0,
            100200.0,
        ],
        latitudes=[
            10.0,
            10.0,
            10.0,
            10.0,
        ],
        longitudes=[
            250.0,
            250.0,
            250.0,
            250.0,
        ],
        lead_times=[
            0,
            6,
            12,
            18,
        ],
    )

    u, v = make_wind(
        [0, 6, 12, 18],
        [
            10.0,
            18.0,
            19.0,
            20.0,
        ],
    )

    results = detect_genesis(
        [track],
        u_wind=u,
        v_wind=v,
        minimum_wind=17.0,
        maximum_pressure=100500.0,
        minimum_consecutive_points=3,
        wind_radius_km=300.0,
    )

    assert len(results) == 1

    result = results[0]

    assert isinstance(
        result,
        GenesisResult,
    )

    assert result.track_index == 0
    assert result.genesis_lead_time_hours == 6
    assert result.pressure == pytest.approx(
        100400.0
    )

    assert result.max_wind == pytest.approx(
        18.0
    )

    assert result.qualifying_points == 3


def test_detect_genesis_rejects_short_run():
    track = make_track(
        pressures=[
            100400.0,
            100300.0,
            101000.0,
        ],
        latitudes=[
            10.0,
            10.0,
            10.0,
        ],
        longitudes=[
            250.0,
            250.0,
            250.0,
        ],
        lead_times=[
            0,
            6,
            12,
        ],
    )

    u, v = make_wind(
        [0, 6, 12],
        [
            18.0,
            19.0,
            5.0,
        ],
    )

    results = detect_genesis(
        [track],
        u_wind=u,
        v_wind=v,
        minimum_wind=17.0,
        maximum_pressure=100500.0,
        minimum_consecutive_points=3,
    )

    assert results == []


def test_detect_genesis_multiple_tracks():
    track1 = make_track(
        pressures=[
            101000.0,
            101000.0,
            101000.0,
        ],
        latitudes=[
            10.0,
            10.0,
            10.0,
        ],
        longitudes=[
            250.0,
            250.0,
            250.0,
        ],
        lead_times=[
            0,
            6,
            12,
        ],
    )

    track2 = make_track(
        pressures=[
            100400.0,
            100300.0,
            100200.0,
        ],
        latitudes=[
            10.0,
            10.0,
            10.0,
        ],
        longitudes=[
            250.0,
            250.0,
            250.0,
        ],
        lead_times=[
            0,
            6,
            12,
        ],
    )

    u, v = make_wind(
        [0, 6, 12],
        [
            18.0,
            19.0,
            20.0,
        ],
    )

    results = detect_genesis(
        [
            track1,
            track2,
        ],
        u_wind=u,
        v_wind=v,
        minimum_wind=17.0,
        maximum_pressure=100500.0,
        minimum_consecutive_points=3,
    )

    assert len(results) == 1
    assert results[0].track_index == 1


def test_select_first_genesis():
    results = [
        GenesisResult(
            track_index=0,
            genesis_lead_time_hours=60,
            latitude=15.0,
            longitude=250.0,
            pressure=100000.0,
            max_wind=18.0,
            qualifying_points=5,
        ),
        GenesisResult(
            track_index=1,
            genesis_lead_time_hours=54,
            latitude=13.0,
            longitude=254.0,
            pressure=100300.0,
            max_wind=17.5,
            qualifying_points=6,
        ),
    ]

    selected = select_first_genesis(
        results
    )

    assert selected is not None
    assert selected.track_index == 1
    assert selected.genesis_lead_time_hours == 54


def test_select_first_genesis_tie_uses_pressure():
    results = [
        GenesisResult(
            track_index=0,
            genesis_lead_time_hours=54,
            latitude=15.0,
            longitude=250.0,
            pressure=100400.0,
            max_wind=18.0,
            qualifying_points=5,
        ),
        GenesisResult(
            track_index=1,
            genesis_lead_time_hours=54,
            latitude=13.0,
            longitude=254.0,
            pressure=100200.0,
            max_wind=17.5,
            qualifying_points=5,
        ),
    ]

    selected = select_first_genesis(
        results
    )

    assert selected is not None
    assert selected.track_index == 1


def test_select_first_genesis_empty():
    assert (
        select_first_genesis([])
        is None
    )


def test_detect_genesis_empty_tracks():
    u, v = make_wind(
        [0],
        [10.0],
    )

    results = detect_genesis(
        [],
        u_wind=u,
        v_wind=v,
        minimum_wind=17.0,
        maximum_pressure=100500.0,
    )

    assert results == []


def test_detect_genesis_invalid_wind_threshold():
    u, v = make_wind(
        [0],
        [10.0],
    )

    with pytest.raises(ValueError):
        detect_genesis(
            [],
            u_wind=u,
            v_wind=v,
            minimum_wind=-1.0,
            maximum_pressure=100500.0,
        )


def test_detect_genesis_invalid_persistence():
    u, v = make_wind(
        [0],
        [10.0],
    )

    with pytest.raises(ValueError):
        detect_genesis(
            [],
            u_wind=u,
            v_wind=v,
            minimum_wind=17.0,
            maximum_pressure=100500.0,
            minimum_consecutive_points=0,
        )


def test_detect_genesis_missing_lead_time():
    track = make_track(
        pressures=[
            100400.0,
        ],
        latitudes=[
            10.0,
        ],
        longitudes=[
            250.0,
        ],
        lead_times=[
            6,
        ],
    )

    u, v = make_wind(
        [0],
        [
            20.0,
        ],
    )

    with pytest.raises(ValueError):
        detect_genesis(
            [track],
            u_wind=u,
            v_wind=v,
            minimum_wind=17.0,
            maximum_pressure=100500.0,
            minimum_consecutive_points=1,
        )
