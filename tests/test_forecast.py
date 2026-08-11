from datetime import datetime

import numpy as np

import pytest

from aiweather.forecast import Forecast, ForecastMetadata, open_forecast


FORECAST_PATH = (
    "outputs/graphcast_gfs_20260724T000000_240h.zarr"
)


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

    # The forecast should remain lazily loaded.
    assert hasattr(forecast.dataset["t2m"].data, "compute")


def test_missing_forecast():
    with pytest.raises(FileNotFoundError):
        open_forecast(
            "outputs/does_not_exist.zarr"
        )


def test_invalid_forecast_path(tmp_path):
    invalid_path = tmp_path / "invalid.zarr"
    invalid_path.mkdir()

    with pytest.raises(ValueError):
        # The store exists, but its name does not follow
        # the AIWeather forecast naming convention.
        open_forecast(invalid_path)
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
