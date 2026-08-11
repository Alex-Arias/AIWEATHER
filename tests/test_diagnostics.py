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
