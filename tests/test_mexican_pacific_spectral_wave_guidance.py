"""Tests for Mexican Pacific spectral-wave guidance."""

import sys
from pathlib import Path

import numpy as np
import pytest
import xarray as xr


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"

sys.path.insert(0, str(SCRIPTS))

from build_mexican_pacific_spectral_wave_guidance import (
    SPECTRAL_BANDS,
    build_spectral_guidance,
    validate_spectral_dataset,
)


ANCHORS = (
    ("Test Coast", "Test Anchor", 20.0, -105.0),
)


def make_dataset():
    """Create a small, fully valid synthetic wave forecast."""

    coords = {
        "time": np.array(
            ["2026-09-28T12:00:00"],
            dtype="datetime64[ns]",
        ),
        "lead_time": np.array(
            [0, 6],
            dtype="timedelta64[h]",
        ),
        "lat": np.array(
            [19.75, 20.0, 20.25],
        ),
        "lon": np.array(
            [254.75, 255.0, 255.25],
        ),
    }

    shape = (1, 2, 3, 3)

    variables = {
        "swh": 4.0,
        "mwp": 10.0,
        "cos_mwd": 1.0,
        "sin_mwd": 0.0,
    }

    for name in SPECTRAL_BANDS:
        variables[name] = 0.5

    data_vars = {
        name: (
            ("time", "lead_time", "lat", "lon"),
            np.full(shape, value, dtype=float),
        )
        for name, value in variables.items()
    }

    return xr.Dataset(
        data_vars,
        coords=coords,
    )


def test_complete_dataset():

    ds = make_dataset()

    result = build_spectral_guidance(
        ds,
        anchors=ANCHORS,
    )

    assert len(result) == 2

    assert result.spectral_sampling_supported.all()

    assert (
        result.n_missing_spectral_cells == 0
    ).all()

    assert np.allclose(
        result.swh_mean_m,
        4.0,
    )

    expected_fraction = (
        6 * 0.5**2
    ) / 4.0**2

    assert np.allclose(
        result.band_energy_fraction,
        expected_fraction,
    )


def test_original_mask_preserved():

    ds = make_dataset()

    # Missing MWP excludes a cell from the original mask.
    ds["mwp"].values[0, 0, 1, 1] = np.nan

    result = build_spectral_guidance(
        ds,
        anchors=ANCHORS,
    )

    first = result.iloc[0]

    assert first.n_original_valid_cells == 8
    assert first.n_spectral_valid_cells == 8

    assert first.n_missing_spectral_cells == 0


def test_missing_spectral_cell():

    ds = make_dataset()

    ds["h1417"].values[0, 0, 1, 1] = np.nan

    result = build_spectral_guidance(
        ds,
        anchors=ANCHORS,
    )

    first = result.iloc[0]

    assert first.n_original_valid_cells == 9
    assert first.n_spectral_valid_cells == 8

    assert first.n_missing_spectral_cells == 1


def test_missing_required_band():

    ds = make_dataset().drop_vars("h1721")

    with pytest.raises(
        ValueError,
        match="Missing spectral variables",
    ):
        validate_spectral_dataset(ds)


def test_invalid_radius():

    ds = make_dataset()

    with pytest.raises(
        ValueError,
        match="radius_km",
    ):
        build_spectral_guidance(
            ds,
            radius_km=0,
            anchors=ANCHORS,
        )


def test_energy_excess_diagnostic():

    ds = make_dataset()

    # Six bands with Hs=2 m each:
    # combined squared height = 24 m².
    # Total Hs² = 16 m².
    for name in SPECTRAL_BANDS:
        ds[name].values[:] = 2.0

    result = build_spectral_guidance(
        ds,
        anchors=ANCHORS,
    )

    assert np.allclose(
        result.band_energy_fraction,
        1.5,
    )

    assert np.allclose(
        result.fraction_cells_energy_excess,
        1.0,
    )


def test_missing_all_spectral_cells():

    ds = make_dataset()

    ds["h1012"].values[:] = np.nan

    result = build_spectral_guidance(
        ds,
        anchors=ANCHORS,
    )

    assert not result.spectral_sampling_supported.any()

    assert result.band_energy_fraction.isna().all()

    assert (
        result.n_missing_spectral_cells
        == result.n_original_valid_cells
    ).all()
