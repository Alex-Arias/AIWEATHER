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
