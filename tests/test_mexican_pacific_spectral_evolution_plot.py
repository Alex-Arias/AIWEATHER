"""Tests for Mexican Pacific spectral evolution plotting."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from plot_mexican_pacific_spectral_evolution import (
    BANDS,
    SECTORS,
    plot_spectral_evolution,
    validate_input,
)


def make_dataframe():
    """Create a complete synthetic coastal spectral forecast."""

    rows = []

    for sector in SECTORS:
        for lead in range(0, 241, 6):

            row = {
                "sector": sector,
                "lead_time_hours": lead,
                "swh_mean_m": 2.0,
                "band_energy_fraction": 0.25,
                "n_original_valid_cells": 10,
                "n_spectral_valid_cells": 10,
            }

            for band, _ in BANDS:
                row[f"{band}_mean_m"] = 0.3

            rows.append(row)

    return pd.DataFrame(rows)


def test_complete_input():

    df = make_dataframe()

    validate_input(df)


def test_missing_band():

    df = make_dataframe()

    df = df.drop(columns=["h1417_mean_m"])

    with pytest.raises(ValueError, match="Missing columns"):
        validate_input(df)


def test_missing_forecast_lead():

    df = make_dataframe()

    df = df.loc[
        ~(
            (df.sector == "Colima")
            & (df.lead_time_hours == 60)
        )
    ]

    with pytest.raises(ValueError, match="41 forecast records"):
        validate_input(df)


def test_nonfinite_values():

    df = make_dataframe()

    df.loc[
        df.sector == "Sinaloa",
        "h1012_mean_m",
    ] = np.nan

    with pytest.raises(ValueError, match="non-finite"):
        validate_input(df)


def test_sampling_mask_mismatch():

    df = make_dataframe()

    df.loc[
        df.sector == "Nayarit",
        "n_spectral_valid_cells",
    ] = 9

    with pytest.raises(ValueError, match="masks differ"):
        validate_input(df)


def test_plot_output(tmp_path):

    df = make_dataframe()

    prefix = tmp_path / "spectral_evolution"

    plot_spectral_evolution(df, prefix)

    png = prefix.with_suffix(".png")
    pdf = prefix.with_suffix(".pdf")

    assert png.is_file()
    assert pdf.is_file()

    assert png.stat().st_size > 0
    assert pdf.stat().st_size > 0
