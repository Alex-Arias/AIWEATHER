"""
Tests for the ERA5Dataset implementation.
"""

from pathlib import Path

from aiweather.datasets import ERA5Dataset


TEST_DATA = Path("tests/data/era5_sample.nc")


def test_open_dataset():
    ds = ERA5Dataset(TEST_DATA)

    ds.open()

    assert ds.dataset is not None

    ds.close()


def test_variables():
    ds = ERA5Dataset(TEST_DATA)

    ds.open()

    variables = ds.variables()

    assert "t2m" in variables
    assert "u10" in variables
    assert "v10" in variables

    ds.close()


def test_times():
    ds = ERA5Dataset(TEST_DATA)

    ds.open()

    times = ds.times()

    assert len(times) == 2

    ds.close()


def test_grid():
    ds = ERA5Dataset(TEST_DATA)

    ds.open()

    grid = ds.grid()

    assert "latitude" in grid
    assert "longitude" in grid

    assert len(grid["latitude"]) == 3
    assert len(grid["longitude"]) == 4

    ds.close()