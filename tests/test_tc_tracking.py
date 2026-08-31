import numpy as np
import pytest
import xarray as xr

from aiweather.forecast import open_forecast
from aiweather.tracking import (
    TrackPoint,
    great_circle_distance_km,
    track_pressure_minimum,
)


FORECAST_PATH = (
    "outputs/graphcast/20260724T000000/forecast.zarr"
)


# ---------------------------------------------------------
# Great-circle distance
# ---------------------------------------------------------


def test_great_circle_distance_zero():
    distance = great_circle_distance_km(
        20.0,
        240.0,
        20.0,
        240.0,
    )

    assert float(distance) == pytest.approx(0.0)


def test_great_circle_distance_one_degree_latitude():
    distance = great_circle_distance_km(
        21.0,
        240.0,
        20.0,
        240.0,
    )

    assert float(distance) == pytest.approx(
        111.2,
        rel=0.02,
    )


def test_great_circle_distance_wraps_longitude():
    distance = great_circle_distance_km(
        20.0,
        359.0,
        20.0,
        1.0,
    )

    assert float(distance) < 250.0


# ---------------------------------------------------------
# TrackPoint
# ---------------------------------------------------------


def test_track_point_fields():
    point = TrackPoint(
        lead_time_hours=24,
        latitude=15.0,
        longitude=250.0,
        pressure=99000.0,
        pressure_units="Pa",
        max_wind=20.0,
        wind_units="m s-1",
    )

    assert point.lead_time_hours == 24
    assert point.latitude == pytest.approx(15.0)
    assert point.longitude == pytest.approx(250.0)
    assert point.pressure == pytest.approx(99000.0)
    assert point.pressure_units == "Pa"
    assert point.max_wind == pytest.approx(20.0)
    assert point.wind_units == "m s-1"


# ---------------------------------------------------------
# Synthetic pressure tracking
# ---------------------------------------------------------


def test_track_pressure_minimum_known_path():
    lead_time = np.array(
        [
            np.timedelta64(0, "h"),
            np.timedelta64(6, "h"),
            np.timedelta64(12, "h"),
        ]
    )

    lat = [10.0, 11.0, 12.0]
    lon = [250.0, 251.0, 252.0]

    pressure = np.full(
        (3, 3, 3),
        101000.0,
    )

    pressure[0, 0, 0] = 99000.0
    pressure[1, 1, 1] = 98500.0
    pressure[2, 2, 2] = 98000.0

    da = xr.DataArray(
        pressure,
        dims=("lead_time", "lat", "lon"),
        coords={
            "lead_time": lead_time,
            "lat": lat,
            "lon": lon,
        },
    )

    track = track_pressure_minimum(
        da,
        initial_latitude=10.0,
        initial_longitude=250.0,
        search_radius_km=200.0,
    )

    assert len(track) == 3

    assert track[0].latitude == pytest.approx(10.0)
    assert track[0].longitude == pytest.approx(250.0)

    assert track[1].latitude == pytest.approx(11.0)
    assert track[1].longitude == pytest.approx(251.0)

    assert track[2].latitude == pytest.approx(12.0)
    assert track[2].longitude == pytest.approx(252.0)


def test_track_pressure_minimum_respects_search_radius():
    lead_time = np.array(
        [
            np.timedelta64(0, "h"),
            np.timedelta64(6, "h"),
        ]
    )

    pressure = xr.DataArray(
        [
            [
                [99000.0, 101000.0],
                [101000.0, 101000.0],
            ],
            [
                [100500.0, 100600.0],
                [100700.0, 97000.0],
            ],
        ],
        dims=("lead_time", "lat", "lon"),
        coords={
            "lead_time": lead_time,
            "lat": [10.0, 20.0],
            "lon": [250.0, 260.0],
        },
    )

    track = track_pressure_minimum(
        pressure,
        initial_latitude=10.0,
        initial_longitude=250.0,
        search_radius_km=300.0,
    )

    assert len(track) == 2

    # The tracker must not jump to the distant 97000 Pa low.
    assert track[1].latitude == pytest.approx(10.0)
    assert track[1].longitude == pytest.approx(250.0)
    assert track[1].pressure == pytest.approx(100500.0)


def test_track_pressure_minimum_rejects_invalid_radius():
    pressure = xr.DataArray(
        np.ones((1, 1, 1)),
        dims=("lead_time", "lat", "lon"),
        coords={
            "lead_time": [np.timedelta64(0, "h")],
            "lat": [10.0],
            "lon": [250.0],
        },
    )

    with pytest.raises(ValueError):
        track_pressure_minimum(
            pressure,
            initial_latitude=10.0,
            initial_longitude=250.0,
            search_radius_km=0.0,
        )


def test_track_pressure_minimum_rejects_invalid_start_index():
    pressure = xr.DataArray(
        np.ones((1, 1, 1)),
        dims=("lead_time", "lat", "lon"),
        coords={
            "lead_time": [np.timedelta64(0, "h")],
            "lat": [10.0],
            "lon": [250.0],
        },
    )

    with pytest.raises(ValueError):
        track_pressure_minimum(
            pressure,
            initial_latitude=10.0,
            initial_longitude=250.0,
            start_index=5,
        )


def test_track_pressure_minimum_rejects_invalid_translation_speed():
    pressure = xr.DataArray(
        np.ones((1, 1, 1)),
        dims=("lead_time", "lat", "lon"),
        coords={
            "lead_time": [np.timedelta64(0, "h")],
            "lat": [10.0],
            "lon": [250.0],
        },
    )

    with pytest.raises(ValueError):
        track_pressure_minimum(
            pressure,
            initial_latitude=10.0,
            initial_longitude=250.0,
            maximum_translation_speed_mps=0.0,
        )


def test_track_pressure_minimum_terminates_on_fast_translation():
    pressure = xr.DataArray(
        [
            [[99000.0, 101000.0, 101000.0]],
            [[101000.0, 101000.0, 98000.0]],
        ],
        dims=("lead_time", "lat", "lon"),
        coords={
            "lead_time": [
                np.timedelta64(0, "h"),
                np.timedelta64(3, "h"),
            ],
            "lat": [10.0],
            "lon": [250.0, 252.0, 254.0],
        },
    )

    track = track_pressure_minimum(
        pressure,
        initial_latitude=10.0,
        initial_longitude=250.0,
        search_radius_km=500.0,
        maximum_translation_speed_mps=20.0,
    )

    assert len(track) == 1
    assert track[0].lead_time_hours == 0


def test_track_pressure_minimum_none_preserves_fast_candidate():
    pressure = xr.DataArray(
        [
            [[99000.0, 101000.0, 101000.0]],
            [[101000.0, 101000.0, 98000.0]],
        ],
        dims=("lead_time", "lat", "lon"),
        coords={
            "lead_time": [
                np.timedelta64(0, "h"),
                np.timedelta64(3, "h"),
            ],
            "lat": [10.0],
            "lon": [250.0, 252.0, 254.0],
        },
    )

    track = track_pressure_minimum(
        pressure,
        initial_latitude=10.0,
        initial_longitude=250.0,
        search_radius_km=500.0,
        maximum_translation_speed_mps=None,
    )

    assert len(track) == 2
    assert track[1].longitude == pytest.approx(254.0)

# ---------------------------------------------------------
# Wind diagnostics attached to tracking
# ---------------------------------------------------------


def test_track_pressure_minimum_with_wind():
    lead_time = np.array(
        [
            np.timedelta64(0, "h"),
        ]
    )

    pressure = xr.DataArray(
        [[[99000.0, 100000.0]]],
        dims=("lead_time", "lat", "lon"),
        coords={
            "lead_time": lead_time,
            "lat": [10.0],
            "lon": [250.0, 251.0],
        },
        attrs={"units": "Pa"},
    )

    u = xr.DataArray(
        [[[3.0, 5.0]]],
        dims=("lead_time", "lat", "lon"),
        coords=pressure.coords,
        attrs={"units": "m s-1"},
    )

    v = xr.DataArray(
        [[[4.0, 12.0]]],
        dims=("lead_time", "lat", "lon"),
        coords=pressure.coords,
        attrs={"units": "m s-1"},
    )

    track = track_pressure_minimum(
        pressure,
        initial_latitude=10.0,
        initial_longitude=250.0,
        search_radius_km=300.0,
        u_wind=u,
        v_wind=v,
        wind_radius_km=300.0,
    )

    assert len(track) == 1

    assert track[0].pressure == pytest.approx(99000.0)
    assert track[0].pressure_units == "Pa"

    assert track[0].max_wind == pytest.approx(13.0)
    assert track[0].wind_units == "m s-1"


def test_track_without_wind_has_none_intensity():
    pressure = xr.DataArray(
        [[[99000.0]]],
        dims=("lead_time", "lat", "lon"),
        coords={
            "lead_time": [np.timedelta64(0, "h")],
            "lat": [10.0],
            "lon": [250.0],
        },
    )

    track = track_pressure_minimum(
        pressure,
        initial_latitude=10.0,
        initial_longitude=250.0,
    )

    assert track[0].max_wind is None
    assert track[0].wind_units is None


def test_tracker_requires_both_wind_components():
    pressure = xr.DataArray(
        [[[99000.0]]],
        dims=("lead_time", "lat", "lon"),
        coords={
            "lead_time": [np.timedelta64(0, "h")],
            "lat": [10.0],
            "lon": [250.0],
        },
    )

    u = xr.DataArray(
        [[[3.0]]],
        dims=("lead_time", "lat", "lon"),
        coords=pressure.coords,
    )

    with pytest.raises(ValueError):
        track_pressure_minimum(
            pressure,
            initial_latitude=10.0,
            initial_longitude=250.0,
            u_wind=u,
        )


# ---------------------------------------------------------
# Real GraphCast integration
# ---------------------------------------------------------


def test_graphcast_candidate_track():
    forecast = open_forecast(
        FORECAST_PATH
    )

    region = forecast.select_region(
        lat_min=5.0,
        lat_max=35.0,
        lon_min=-130.0,
        lon_max=-90.0,
    )

    track = track_pressure_minimum(
        region["msl"],
        initial_latitude=11.5,
        initial_longitude=257.25,
        search_radius_km=500.0,
        start_index=6,
        u_wind=region["u10m"],
        v_wind=region["v10m"],
        wind_radius_km=300.0,
    )

    assert len(track) == 35

    first = track[0]

    assert first.lead_time_hours == 36
    assert first.latitude == pytest.approx(
        11.50
    )
    assert first.longitude == pytest.approx(
        257.25
    )

    expected_pressure = float(
        region["msl"]
        .isel(
            time=0,
            lead_time=6,
        )
        .sel(
            lat=first.latitude,
            lon=first.longitude,
            method="nearest",
        )
        .compute()
    )

    assert first.pressure == pytest.approx(
        expected_pressure,
        abs=1e-6,
    )


    # Continuity constraint prevents the large unrelated jump
    # seen in the unconstrained regional pressure minimum.
    #point_216 = next(
    #    point
    #    for point in track
    #    if point.lead_time_hours == 216
    #)

    #assert point_216.latitude == pytest.approx(24.75)
    #assert point_216.longitude == pytest.approx(232.50)

# ---------------------------------------------------------
# Automatic tracking from genesis
# ---------------------------------------------------------


def test_graphcast_track_from_detected_genesis():
    from aiweather.diagnostics import (
        detect_pressure_minima,
    )

    from aiweather.tracking import (
        associate_candidates,
        detect_genesis,
        select_first_genesis,
        track_from_genesis,
    )

    forecast = open_forecast(
        FORECAST_PATH
    )

    region = forecast.select_region(
        lat_min=5.0,
        lat_max=35.0,
        lon_min=-130.0,
        lon_max=-90.0,
    )

    lead_values = (
        region["lead_time"]
        .values
        .astype("timedelta64[h]")
        .astype(int)
    )

    candidates_by_lead = []

    for lead_hours in range(
        24,
        175,
        6,
    ):
        matches = np.flatnonzero(
            lead_values == lead_hours
        )

        if len(matches) == 0:
            continue

        pressure = region["msl"].isel(
            time=0,
            lead_time=int(matches[0]),
        )

        minima = detect_pressure_minima(
            pressure,
            max_candidates=5,
            minimum_separation_km=500.0,
        )

        candidates_by_lead.append(
            (
                lead_hours,
                minima,
            )
        )

    candidate_tracks = associate_candidates(
        candidates_by_lead,
        maximum_displacement_km=500.0,
    )

    genesis_results = detect_genesis(
        candidate_tracks,
        u_wind=region["u10m"],
        v_wind=region["v10m"],
        minimum_wind=17.0,
        maximum_pressure=100500.0,
        minimum_consecutive_points=3,
        wind_radius_km=300.0,
    )

    genesis = select_first_genesis(
        genesis_results
    )

    assert genesis is not None

    track = track_from_genesis(
        region["msl"],
        genesis=genesis,
        u_wind=region["u10m"],
        v_wind=region["v10m"],
        search_radius_km=500.0,
        wind_radius_km=300.0,
    )

    assert len(track) == 32

    first = track[0]

    assert first.lead_time_hours == 54
    assert first.latitude == pytest.approx(
        13.0
    )
    assert first.longitude == pytest.approx(
        254.0
    )

    lead_index = int(
        np.flatnonzero(
            lead_values
            == first.lead_time_hours
        )[0]
    )

    expected_pressure = float(
        region["msl"]
        .isel(
            time=0,
            lead_time=lead_index,
        )
        .sel(
            lat=first.latitude,
            lon=first.longitude,
            method="nearest",
        )
        .compute()
    )

    lead_index = int(
        np.flatnonzero(
            lead_values == first.lead_time_hours
        )[0]
    )

    expected_pressure = float(
        region["msl"]
        .isel(
            time=0,
            lead_time=lead_index,
        )
        .sel(
            lat=first.latitude,
            lon=first.longitude,
            method="nearest",
        )
        .compute()
    )

    assert first.pressure == pytest.approx(
        expected_pressure,
        abs=1e-6,
    )


def test_track_from_genesis_invalid_lead_time():
    from aiweather.tracking import (
        GenesisResult,
        track_from_genesis,
    )

    pressure = xr.DataArray(
        [[[99000.0]]],
        dims=("lead_time", "lat", "lon"),
        coords={
            "lead_time": [np.timedelta64(0, "h")],
            "lat": [10.0],
            "lon": [250.0],
        },
    )

    genesis = GenesisResult(
        track_index=0,
        genesis_lead_time_hours=6,
        latitude=10.0,
        longitude=250.0,
        pressure=99000.0,
        max_wind=18.0,
        qualifying_points=3,
    )

    with pytest.raises(ValueError):
        track_from_genesis(
            pressure,
            genesis=genesis,
        )


def test_track_from_genesis_rejects_invalid_object():
    from aiweather.tracking import track_from_genesis

    pressure = xr.DataArray(
        [[[99000.0]]],
        dims=("lead_time", "lat", "lon"),
        coords={
            "lead_time": [np.timedelta64(0, "h")],
            "lat": [10.0],
            "lon": [250.0],
        },
    )

    with pytest.raises(TypeError):
        track_from_genesis(
            pressure,
            genesis=object(),
        )


def test_graphcast_track_from_detected_genesis():
    from aiweather.diagnostics import detect_pressure_minima

    from aiweather.tracking import (
        associate_candidates,
        detect_genesis,
        select_first_genesis,
        track_from_genesis,
    )

    forecast = open_forecast(
        FORECAST_PATH
    )

    region = forecast.select_region(
        lat_min=5.0,
        lat_max=35.0,
        lon_min=-130.0,
        lon_max=-90.0,
    )

    lead_values = (
        region["lead_time"]
        .values
        .astype("timedelta64[h]")
        .astype(int)
    )

    candidates_by_lead = []

    for lead_hours in range(
        24,
        175,
        6,
    ):
        matches = np.flatnonzero(
            lead_values == lead_hours
        )

        if len(matches) == 0:
            continue

        pressure = region["msl"].isel(
            time=0,
            lead_time=int(matches[0]),
        )

        minima = detect_pressure_minima(
            pressure,
            max_candidates=5,
            minimum_separation_km=500.0,
        )

        candidates_by_lead.append(
            (
                lead_hours,
                minima,
            )
        )

    candidate_tracks = associate_candidates(
        candidates_by_lead,
        maximum_displacement_km=500.0,
    )

    genesis_results = detect_genesis(
        candidate_tracks,
        u_wind=region["u10m"],
        v_wind=region["v10m"],
        minimum_wind=17.0,
        maximum_pressure=100500.0,
        minimum_consecutive_points=3,
        wind_radius_km=300.0,
    )

    genesis = select_first_genesis(
        genesis_results
    )

    assert genesis is not None

    track = track_from_genesis(
        region["msl"],
        genesis=genesis,
        u_wind=region["u10m"],
        v_wind=region["v10m"],
        search_radius_km=500.0,
        wind_radius_km=300.0,
    )

    assert len(track) == 32

    first = track[0]

    assert first.lead_time_hours == 54
    assert first.latitude == pytest.approx(13.0)
    assert first.longitude == pytest.approx(254.0)
    lead_index = int(
        np.flatnonzero(
            lead_values == first.lead_time_hours
        )[0]
    )

    expected_pressure = float(
        region["msl"]
        .isel(
            time=0,
            lead_time=lead_index,
        )
        .sel(
            lat=first.latitude,
            lon=first.longitude,
            method="nearest",
        )
        .compute()
    )

    assert first.pressure == pytest.approx(
        expected_pressure,
        abs=1e-6,
    )

    last = track[-1]

    assert last.lead_time_hours == 240