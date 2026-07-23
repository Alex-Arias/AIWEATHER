"""
Tests for the AIWeather IO backend.
"""

import pytest

from aiweather.io import detect_engine


def test_detect_netcdf():
    assert detect_engine("era5.nc") == "netcdf4"


def test_detect_zarr():
    assert detect_engine("forecast.zarr") == "zarr"


def test_detect_grib():
    assert detect_engine("gfs.grib2") == "cfgrib"


def test_unknown_extension():
    with pytest.raises(ValueError):
        detect_engine("forecast.txt")