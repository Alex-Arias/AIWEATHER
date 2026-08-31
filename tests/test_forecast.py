from datetime import datetime

import numpy as np
import pytest
import xarray as xr

from aiweather.forecast import (
    Forecast,
    ForecastMetadata,
    open_forecast,
)


FORECAST_PATH = (
    "outputs/graphcast/20260724T000000/forecast.zarr"
)


# ---------------------------------------------------------
# Forecast loading
# ---------------------------------------------------------


def test_open_forecast():
    forecast = open_forecast(FORECAST_PATH)

    assert isinstance(forecast, Forecast)


def test_forecast_metadata():
    forecast = open_forecast(FORECAST_PATH)

    metadata = forecast.metadata

    assert isinstance(metadata, ForecastMetadata)
    assert metadata.model_name == "graphcast"
    assert metadata.backend == "earth2studio"
    assert metadata.forecast_id == (
        "graphcast_gfs_20260724T000000_240h"
    )
    assert metadata.initialization_time == datetime(
        2026, 7, 24, 0, 0
    )


def test_forecast_dimensions():
    forecast = open_forecast(FORECAST_PATH)

    assert forecast.shape["time"] == 1
    assert forecast.shape["lead_time"] == 41
    assert forecast.shape["lat"] == 721
    assert forecast.shape["lon"] == 1440


def test_forecast_variables():
    forecast = open_forecast(FORECAST_PATH)

    assert "t2m" in forecast.variables
    assert "msl" in forecast.variables
    assert "u10m" in forecast.variables
    assert "v10m" in forecast.variables


def test_forecast_coordinates():
    forecast = open_forecast(FORECAST_PATH)

    assert forecast.latitude is not None
    assert forecast.longitude is not None
    assert forecast.lead_time is not None


def test_forecast_initialization_time():
    forecast = open_forecast(FORECAST_PATH)

    assert forecast.initialization_time is not None
    assert forecast.initialization_time.values[0] == (
        np.datetime64("2026-07-24T00:00:00.000000000")
    )


def test_forecast_to_xarray():
    forecast = open_forecast(FORECAST_PATH)

    dataset = forecast.to_xarray()

    assert "t2m" in dataset
    assert dataset.sizes["lead_time"] == 41


def test_forecast_is_lazy():
    forecast = open_forecast(FORECAST_PATH)

    assert hasattr(
        forecast.dataset["t2m"].data,
        "compute",
    )


def test_missing_forecast():
    with pytest.raises(FileNotFoundError):
        open_forecast(
            "outputs/does_not_exist.zarr"
        )


def test_invalid_forecast_path(tmp_path):
    invalid_path = tmp_path / "invalid.zarr"
    invalid_path.mkdir()

    with pytest.raises(ValueError):
        open_forecast(invalid_path)

def test_open_forecast_noncanonical_name_with_metadata(
    tmp_path,
):
    path = tmp_path / "forecast_azure.zarr"

    dataset = xr.Dataset(
        data_vars={
            "msl": (
                (
                    "time",
                    "lead_time",
                    "lat",
                    "lon",
                ),
                np.ones(
                    (1, 2, 2, 2),
                    dtype=np.float32,
                ),
            ),
        },
        coords={
            "time": [
                np.datetime64(
                    "2026-08-29T12:00:00"
                )
            ],
            "lead_time": np.array(
                [
                    np.timedelta64(0, "h"),
                    np.timedelta64(6, "h"),
                ]
            ),
            "lat": [15.0, 16.0],
            "lon": [242.0, 243.0],
        },
        attrs={
            "aiweather_model": "aifs2",
            "aiweather_forecast_id": (
                "aifs2_ifs_"
                "20260829T120000_6h"
            ),
            "aiweather_initialization_time": (
                "2026-08-29T12:00:00"
            ),
            "aiweather_backend": "earth2studio",
            "aiweather_datasource": "ifs",
        },
    )

    dataset.to_zarr(
        path,
        mode="w",
    )

    forecast = open_forecast(path)

    assert isinstance(forecast, Forecast)
    assert forecast.metadata.model_name == "aifs2"
    assert (
        forecast.metadata.forecast_id
        == "aifs2_ifs_20260829T120000_6h"
    )
    assert (
        forecast.metadata.initialization_time
        == datetime(2026, 8, 29, 12)
    )
    assert (
        forecast.metadata.backend
        == "earth2studio"
    )

# ---------------------------------------------------------
# Variable access
# ---------------------------------------------------------


def test_get_variable():
    forecast = open_forecast(FORECAST_PATH)

    variable = forecast.get_variable("t2m")

    assert variable.name == "t2m"
    assert "lead_time" in variable.dims
    assert hasattr(variable.data, "compute")


def test_get_variable_at_lead_time():
    forecast = open_forecast(FORECAST_PATH)

    variable = forecast.get_variable(
        "t2m",
        lead_time=24,
    )

    assert variable.name == "t2m"
    assert variable.sizes["lead_time"] == 1
    assert variable.lead_time.values[0] == (
        np.timedelta64(24, "h")
    )


def test_get_variable_missing():
    forecast = open_forecast(FORECAST_PATH)

    with pytest.raises(KeyError):
        forecast.get_variable("not_a_real_variable")


def test_get_variable_invalid_lead_time():
    forecast = open_forecast(FORECAST_PATH)

    with pytest.raises(ValueError):
        forecast.get_variable(
            "t2m",
            lead_time=999,
        )


# ---------------------------------------------------------
# Lead-time selection
# ---------------------------------------------------------


def test_select_lead_time():
    forecast = open_forecast(FORECAST_PATH)

    selected = forecast.select_lead_time(24)

    assert isinstance(selected, Forecast)
    assert selected.shape["lead_time"] == 1
    assert selected.lead_time.values[0] == (
        np.timedelta64(24, "h")
    )

    # Original forecast remains unchanged.
    assert forecast.shape["lead_time"] == 41


def test_select_lead_time_invalid():
    forecast = open_forecast(FORECAST_PATH)

    with pytest.raises(ValueError):
        forecast.select_lead_time(999)


def test_selection_preserves_metadata():
    forecast = open_forecast(FORECAST_PATH)

    selected = forecast.select_lead_time(48)

    assert selected.metadata == forecast.metadata


def test_selection_remains_lazy():
    forecast = open_forecast(FORECAST_PATH)

    selected = forecast.select_lead_time(24)

    assert hasattr(
        selected.dataset["t2m"].data,
        "compute",
    )


# ---------------------------------------------------------
# Valid-time selection
# ---------------------------------------------------------


def test_forecast_valid_time():
    forecast = open_forecast(FORECAST_PATH)

    valid_time = forecast.valid_time

    assert valid_time.dims == ("lead_time",)
    assert valid_time.sizes["lead_time"] == 41

    assert valid_time.values[0] == np.datetime64(
        "2026-07-24T00:00:00.000000000"
    )

    assert valid_time.values[-1] == np.datetime64(
        "2026-08-03T00:00:00.000000000"
    )


def test_select_valid_time():
    forecast = open_forecast(FORECAST_PATH)

    selected = forecast.select_valid_time(
        "2026-07-25T00:00:00"
    )

    assert isinstance(selected, Forecast)
    assert selected.shape["lead_time"] == 1

    assert selected.lead_time.values[0] == (
        np.timedelta64(24, "h")
    )


def test_select_valid_time_datetime():
    forecast = open_forecast(FORECAST_PATH)

    selected = forecast.select_valid_time(
        datetime(2026, 7, 26, 0, 0)
    )

    assert selected.shape["lead_time"] == 1

    assert selected.lead_time.values[0] == (
        np.timedelta64(48, "h")
    )


def test_select_valid_time_numpy_datetime64():
    forecast = open_forecast(FORECAST_PATH)

    selected = forecast.select_valid_time(
        np.datetime64("2026-07-27T00:00:00")
    )

    assert selected.shape["lead_time"] == 1

    assert selected.lead_time.values[0] == (
        np.timedelta64(72, "h")
    )


def test_select_valid_time_invalid():
    forecast = open_forecast(FORECAST_PATH)

    with pytest.raises(ValueError):
        forecast.select_valid_time(
            "2026-08-10T00:00:00"
        )


def test_valid_time_selection_preserves_metadata():
    forecast = open_forecast(FORECAST_PATH)

    selected = forecast.select_valid_time(
        "2026-07-26T00:00:00"
    )

    assert selected.metadata == forecast.metadata


def test_valid_time_selection_remains_lazy():
    forecast = open_forecast(FORECAST_PATH)

    selected = forecast.select_valid_time(
        "2026-07-25T00:00:00"
    )

    assert hasattr(
        selected.dataset["t2m"].data,
        "compute",
    )


def test_valid_time_matches_lead_time():
    forecast = open_forecast(FORECAST_PATH)

    selected = forecast.select_valid_time(
        "2026-07-29T00:00:00"
    )

    assert selected.lead_time.values[0] == (
        np.timedelta64(120, "h")
    )


# ---------------------------------------------------------
# Combined forecast selection
# ---------------------------------------------------------


def test_select_by_point_and_valid_time():
    forecast = open_forecast(FORECAST_PATH)

    selected = forecast.select(
        latitude=31.86,
        longitude=-116.67,
        valid_time="2026-07-25T00:00:00",
    )

    assert selected.sizes["time"] == 1
    assert selected.sizes["lead_time"] == 1

    assert selected["lat"].item() == pytest.approx(31.75)
    assert selected["lon"].item() == pytest.approx(243.25)

    assert selected["lead_time"].values[0] == (
        np.timedelta64(24, "h")
    )


def test_select_by_point_and_valid_time_with_variables():
    forecast = open_forecast(FORECAST_PATH)

    selected = forecast.select(
        latitude=31.86,
        longitude=-116.67,
        valid_time="2026-07-25T00:00:00",
        variables=["t2m", "msl", "u10m", "v10m"],
    )

    assert set(selected.data_vars) == {
        "t2m",
        "msl",
        "u10m",
        "v10m",
    }

    assert selected.sizes["lead_time"] == 1


def test_select_accepts_0360_longitude():
    forecast = open_forecast(FORECAST_PATH)

    selected = forecast.select(
        latitude=31.86,
        longitude=243.33,
        valid_time="2026-07-25T00:00:00",
    )

    assert selected["lat"].item() == pytest.approx(31.75)
    assert selected["lon"].item() == pytest.approx(243.25)


def test_select_variable_subset():
    forecast = open_forecast(FORECAST_PATH)

    selected = forecast.select(
        variables=["t2m", "msl"],
    )

    assert set(selected.data_vars) == {
        "t2m",
        "msl",
    }


def test_select_valid_time_only():
    forecast = open_forecast(FORECAST_PATH)

    selected = forecast.select(
        valid_time="2026-07-26T00:00:00",
    )

    assert selected.sizes["lead_time"] == 1
    assert selected["lead_time"].values[0] == (
        np.timedelta64(48, "h")
    )


def test_select_preserves_lazy_loading():
    forecast = open_forecast(FORECAST_PATH)

    selected = forecast.select(
        latitude=31.86,
        longitude=-116.67,
        valid_time="2026-07-25T00:00:00",
        variables=["t2m"],
    )

    assert hasattr(
        selected["t2m"].data,
        "compute",
    )


def test_select_invalid_valid_time():
    forecast = open_forecast(FORECAST_PATH)

    with pytest.raises(ValueError):
        forecast.select(
            valid_time="2026-08-10T00:00:00",
        )


def test_select_requires_latitude_and_longitude_together():
    forecast = open_forecast(FORECAST_PATH)

    with pytest.raises(ValueError):
        forecast.select(
            latitude=31.86,
        )


def test_select_requires_selection_criteria():
    forecast = open_forecast(FORECAST_PATH)

    with pytest.raises(ValueError):
        forecast.select()


def test_select_missing_variable():
    forecast = open_forecast(FORECAST_PATH)

    with pytest.raises(KeyError):
        forecast.select(
            variables=["t2m", "not_a_real_variable"],
        )





# ---------------------------------------------------------
# Spatial point selection
# ---------------------------------------------------------


def test_select_point_normalizes_negative_longitude():
    forecast = open_forecast(FORECAST_PATH)

    point = forecast.select_point(
        latitude=31.86,
        longitude=-116.67,
    )

    assert point.sizes["time"] == 1
    assert point.sizes["lead_time"] == 41

    assert point["lat"].item() == pytest.approx(31.75)
    assert point["lon"].item() == pytest.approx(243.25)


def test_select_point_accepts_0360_longitude():
    forecast = open_forecast(FORECAST_PATH)

    point = forecast.select_point(
        latitude=31.86,
        longitude=243.33,
    )

    assert point["lat"].item() == pytest.approx(31.75)
    assert point["lon"].item() == pytest.approx(243.25)


def test_select_point_negative_and_0360_are_equivalent():
    forecast = open_forecast(FORECAST_PATH)

    point_negative = forecast.select_point(
        latitude=31.86,
        longitude=-116.67,
    )

    point_360 = forecast.select_point(
        latitude=31.86,
        longitude=243.33,
    )

    assert point_negative["lat"].item() == pytest.approx(
        point_360["lat"].item()
    )

    assert point_negative["lon"].item() == pytest.approx(
        point_360["lon"].item()
    )


def test_select_point_preserves_lazy_loading():
    forecast = open_forecast(FORECAST_PATH)

    point = forecast.select_point(
        latitude=31.86,
        longitude=-116.67,
    )

    assert hasattr(
        point["t2m"].data,
        "compute",
    )


def test_select_point_invalid_latitude():
    forecast = open_forecast(FORECAST_PATH)

    with pytest.raises(ValueError):
        forecast.select_point(
            latitude=95.0,
            longitude=-116.67,
        )


def test_select_point_invalid_longitude():
    forecast = open_forecast(FORECAST_PATH)

    with pytest.raises(ValueError):
        forecast.select_point(
            latitude=31.86,
            longitude=400.0,
        )

# ---------------------------------------------------------
# Regional selection
# ---------------------------------------------------------


def test_select_region_negative_longitude():
    forecast = open_forecast(FORECAST_PATH)

    region = forecast.select_region(
        lat_min=20.0,
        lat_max=35.0,
        lon_min=-120.0,
        lon_max=-105.0,
    )

    assert region.sizes["lat"] > 0
    assert region.sizes["lon"] > 0
    assert region.sizes["time"] == 1
    assert region.sizes["lead_time"] == 41

    assert region["lon"].min().item() >= 240.0
    assert region["lon"].max().item() <= 255.0


def test_select_region_0360_longitude():
    forecast = open_forecast(FORECAST_PATH)

    region = forecast.select_region(
        lat_min=20.0,
        lat_max=35.0,
        lon_min=240.0,
        lon_max=255.0,
    )

    assert region.sizes["lat"] > 0
    assert region.sizes["lon"] > 0
    assert region.sizes["time"] == 1
    assert region.sizes["lead_time"] == 41


def test_select_region_negative_and_0360_are_equivalent():
    forecast = open_forecast(FORECAST_PATH)

    region_negative = forecast.select_region(
        lat_min=20.0,
        lat_max=35.0,
        lon_min=-120.0,
        lon_max=-105.0,
    )

    region_360 = forecast.select_region(
        lat_min=20.0,
        lat_max=35.0,
        lon_min=240.0,
        lon_max=255.0,
    )

    np.testing.assert_array_equal(
        region_negative["lat"].values,
        region_360["lat"].values,
    )

    np.testing.assert_array_equal(
        region_negative["lon"].values,
        region_360["lon"].values,
    )


def test_select_region_handles_descending_latitude():
    forecast = open_forecast(FORECAST_PATH)

    region = forecast.select_region(
        lat_min=20.0,
        lat_max=35.0,
        lon_min=-120.0,
        lon_max=-105.0,
    )

    assert region["lat"].values[0] > region["lat"].values[-1]


def test_select_region_invalid_latitude():
    forecast = open_forecast(FORECAST_PATH)

    with pytest.raises(ValueError):
        forecast.select_region(
            lat_min=-95.0,
            lat_max=35.0,
            lon_min=-120.0,
            lon_max=-105.0,
        )


def test_select_region_invalid_latitude_order():
    forecast = open_forecast(FORECAST_PATH)

    with pytest.raises(ValueError):
        forecast.select_region(
            lat_min=35.0,
            lat_max=20.0,
            lon_min=-120.0,
            lon_max=-105.0,
        )


def test_select_region_invalid_longitude():
    forecast = open_forecast(FORECAST_PATH)

    with pytest.raises(ValueError):
        forecast.select_region(
            lat_min=20.0,
            lat_max=35.0,
            lon_min=-400.0,
            lon_max=-105.0,
        )


def test_select_region_preserves_lazy_loading():
    forecast = open_forecast(FORECAST_PATH)

    region = forecast.select_region(
        lat_min=20.0,
        lat_max=35.0,
        lon_min=-120.0,
        lon_max=-105.0,
    )

    assert hasattr(
        region["t2m"].data,
        "compute",
    )


def test_select_region_does_not_modify_original_forecast():
    forecast = open_forecast(FORECAST_PATH)

    original_shape = dict(forecast.shape)

    forecast.select_region(
        lat_min=20.0,
        lat_max=35.0,
        lon_min=-120.0,
        lon_max=-105.0,
    )

    assert dict(forecast.shape) == original_shape