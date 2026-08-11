import numpy as np
import pytest
import xarray as xr

from aiweather.diagnostics import wind_speed
from aiweather.forecast import open_forecast


FORECAST_PATH = (
    "outputs/graphcast_gfs_20260724T000000_240h.zarr"
)


def test_wind_speed_known_values():
    u = xr.DataArray(
        [3.0, 5.0],
        dims=("point",),
    )

    v = xr.DataArray(
        [4.0, 12.0],
        dims=("point",),
    )

    speed = wind_speed(u, v)

    np.testing.assert_allclose(
        speed.values,
        [5.0, 13.0],
    )


def test_wind_speed_preserves_dimensions():
    u = xr.DataArray(
        np.ones((2, 3)),
        dims=("lat", "lon"),
        coords={
            "lat": [20.0, 21.0],
            "lon": [240.0, 241.0, 242.0],
        },
    )

    v = xr.DataArray(
        np.ones((2, 3)),
        dims=("lat", "lon"),
        coords={
            "lat": [20.0, 21.0],
            "lon": [240.0, 241.0, 242.0],
        },
    )

    speed = wind_speed(u, v)

    assert speed.dims == ("lat", "lon")
    assert speed.sizes == u.sizes

    np.testing.assert_array_equal(
        speed["lat"].values,
        u["lat"].values,
    )

    np.testing.assert_array_equal(
        speed["lon"].values,
        u["lon"].values,
    )


def test_wind_speed_name():
    u = xr.DataArray([3.0], dims=("point",))
    v = xr.DataArray([4.0], dims=("point",))

    speed = wind_speed(u, v)

    assert speed.name == "wind_speed"


def test_wind_speed_custom_name():
    u = xr.DataArray([3.0], dims=("point",))
    v = xr.DataArray([4.0], dims=("point",))

    speed = wind_speed(
        u,
        v,
        name="wind10m",
    )

    assert speed.name == "wind10m"


def test_wind_speed_preserves_matching_units():
    u = xr.DataArray(
        [3.0],
        dims=("point",),
        attrs={"units": "m s-1"},
    )

    v = xr.DataArray(
        [4.0],
        dims=("point",),
        attrs={"units": "m s-1"},
    )

    speed = wind_speed(u, v)

    assert speed.attrs["units"] == "m s-1"


def test_wind_speed_does_not_invent_units():
    u = xr.DataArray([3.0], dims=("point",))
    v = xr.DataArray([4.0], dims=("point",))

    speed = wind_speed(u, v)

    assert "units" not in speed.attrs


def test_wind_speed_rejects_mismatched_coordinates():
    u = xr.DataArray(
        [3.0, 4.0],
        dims=("lat",),
        coords={"lat": [20.0, 21.0]},
    )

    v = xr.DataArray(
        [3.0, 4.0],
        dims=("lat",),
        coords={"lat": [20.0, 22.0]},
    )

    with pytest.raises(ValueError):
        wind_speed(u, v)


def test_wind_speed_rejects_non_dataarray():
    u = np.array([3.0])
    v = xr.DataArray([4.0], dims=("point",))

    with pytest.raises(TypeError):
        wind_speed(u, v)


def test_graphcast_wind_speed_remains_lazy():
    forecast = open_forecast(FORECAST_PATH)

    speed = wind_speed(
        forecast.dataset["u10m"],
        forecast.dataset["v10m"],
    )

    assert speed.dims == (
        "time",
        "lead_time",
        "lat",
        "lon",
    )

    assert speed.shape == (
        1,
        41,
        721,
        1440,
    )

    assert hasattr(
        speed.data,
        "compute",
    )


def test_graphcast_point_wind_speed_value():
    forecast = open_forecast(FORECAST_PATH)

    point = forecast.select_point(
        latitude=20.0,
        longitude=-110.0,
    )

    speed = wind_speed(
        point["u10m"],
        point["v10m"],
    )

    value = speed.isel(
        time=0,
        lead_time=0,
    ).compute()

    expected = np.hypot(
        point["u10m"].isel(
            time=0,
            lead_time=0,
        ).compute(),
        point["v10m"].isel(
            time=0,
            lead_time=0,
        ).compute(),
    )

    assert float(value) == pytest.approx(
        float(expected)
    )

# ---------------------------------------------------------
# Wind direction
# ---------------------------------------------------------


@pytest.mark.parametrize(
    ("u", "v", "expected"),
    [
        (0.0, -1.0, 0.0),    # from north
        (-1.0, 0.0, 90.0),   # from east
        (0.0, 1.0, 180.0),   # from south
        (1.0, 0.0, 270.0),   # from west
    ],
)
def test_wind_direction_cardinal_directions(
    u,
    v,
    expected,
):
    from aiweather.diagnostics import wind_direction

    u_da = xr.DataArray([u], dims=("point",))
    v_da = xr.DataArray([v], dims=("point",))

    direction = wind_direction(u_da, v_da)

    assert direction.item() == pytest.approx(expected)


def test_wind_direction_range():
    from aiweather.diagnostics import wind_direction

    u = xr.DataArray(
        [1.0, -1.0, 1.0, -1.0],
        dims=("point",),
    )

    v = xr.DataArray(
        [1.0, 1.0, -1.0, -1.0],
        dims=("point",),
    )

    direction = wind_direction(u, v)

    assert bool((direction >= 0.0).all())
    assert bool((direction < 360.0).all())


def test_wind_direction_metadata():
    from aiweather.diagnostics import wind_direction

    u = xr.DataArray([3.0], dims=("point",))
    v = xr.DataArray([4.0], dims=("point",))

    direction = wind_direction(u, v)

    assert direction.name == "wind_direction"
    assert direction.attrs["units"] == "degrees"


def test_wind_direction_rejects_mismatched_coordinates():
    from aiweather.diagnostics import wind_direction

    u = xr.DataArray(
        [1.0, 2.0],
        dims=("lat",),
        coords={"lat": [20.0, 21.0]},
    )

    v = xr.DataArray(
        [1.0, 2.0],
        dims=("lat",),
        coords={"lat": [20.0, 22.0]},
    )

    with pytest.raises(ValueError):
        wind_direction(u, v)


def test_graphcast_wind_direction_remains_lazy():
    from aiweather.diagnostics import wind_direction

    forecast = open_forecast(FORECAST_PATH)

    direction = wind_direction(
        forecast.dataset["u10m"],
        forecast.dataset["v10m"],
    )

    assert direction.dims == (
        "time",
        "lead_time",
        "lat",
        "lon",
    )

    assert direction.shape == (
        1,
        41,
        721,
        1440,
    )

    assert hasattr(direction.data, "compute")



# ---------------------------------------------------------
# Pressure minimum
# ---------------------------------------------------------


def test_pressure_minimum_known_field():
    from aiweather.diagnostics import pressure_minimum

    pressure = xr.DataArray(
        [
            [101000.0, 100500.0, 100800.0],
            [100700.0, 98500.0, 100600.0],
        ],
        dims=("lat", "lon"),
        coords={
            "lat": [20.0, 21.0],
            "lon": [240.0, 241.0, 242.0],
        },
    )

    minimum = pressure_minimum(pressure)

    assert minimum.value == pytest.approx(98500.0)
    assert minimum.latitude == pytest.approx(21.0)
    assert minimum.longitude == pytest.approx(241.0)


def test_pressure_minimum_preserves_units():
    from aiweather.diagnostics import pressure_minimum

    pressure = xr.DataArray(
        [[101000.0, 99000.0]],
        dims=("lat", "lon"),
        coords={
            "lat": [20.0],
            "lon": [240.0, 241.0],
        },
        attrs={"units": "Pa"},
    )

    minimum = pressure_minimum(pressure)

    assert minimum.units == "Pa"


def test_pressure_minimum_does_not_invent_units():
    from aiweather.diagnostics import pressure_minimum

    pressure = xr.DataArray(
        [[101000.0, 99000.0]],
        dims=("lat", "lon"),
        coords={
            "lat": [20.0],
            "lon": [240.0, 241.0],
        },
    )

    minimum = pressure_minimum(pressure)

    assert minimum.units is None


def test_pressure_minimum_accepts_singleton_time_dimensions():
    from aiweather.diagnostics import pressure_minimum

    pressure = xr.DataArray(
        [[[[101000.0, 98000.0]]]],
        dims=("time", "lead_time", "lat", "lon"),
        coords={
            "time": [np.datetime64("2026-07-24")],
            "lead_time": [np.timedelta64(0, "h")],
            "lat": [20.0],
            "lon": [240.0, 241.0],
        },
    )

    minimum = pressure_minimum(pressure)

    assert minimum.value == pytest.approx(98000.0)
    assert minimum.latitude == pytest.approx(20.0)
    assert minimum.longitude == pytest.approx(241.0)


def test_pressure_minimum_rejects_multiple_lead_times():
    from aiweather.diagnostics import pressure_minimum

    pressure = xr.DataArray(
        np.ones((2, 2, 2)),
        dims=("lead_time", "lat", "lon"),
        coords={
            "lead_time": [
                np.timedelta64(0, "h"),
                np.timedelta64(6, "h"),
            ],
            "lat": [20.0, 21.0],
            "lon": [240.0, 241.0],
        },
    )

    with pytest.raises(ValueError):
        pressure_minimum(pressure)


def test_pressure_minimum_rejects_missing_coordinates():
    from aiweather.diagnostics import pressure_minimum

    pressure = xr.DataArray(
        np.ones((2, 2)),
        dims=("y", "x"),
    )

    with pytest.raises(ValueError):
        pressure_minimum(pressure)


def test_pressure_minimum_rejects_all_nan_field():
    from aiweather.diagnostics import pressure_minimum

    pressure = xr.DataArray(
        np.full((2, 2), np.nan),
        dims=("lat", "lon"),
        coords={
            "lat": [20.0, 21.0],
            "lon": [240.0, 241.0],
        },
    )

    with pytest.raises(ValueError):
        pressure_minimum(pressure)


def test_graphcast_regional_pressure_minimum():
    from aiweather.diagnostics import pressure_minimum

    forecast = open_forecast(FORECAST_PATH)

    region = forecast.select_region(
        lat_min=5.0,
        lat_max=35.0,
        lon_min=-130.0,
        lon_max=-90.0,
    )

    pressure = region["msl"].isel(
        time=0,
        lead_time=0,
    )

    minimum = pressure_minimum(pressure)

    assert minimum.value == pytest.approx(
        98127.25
    )

    assert 5.0 <= minimum.latitude <= 35.0
    assert 230.0 <= minimum.longitude <= 270.0