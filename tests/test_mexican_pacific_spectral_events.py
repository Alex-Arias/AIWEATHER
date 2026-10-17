"""Tests for Mexican Pacific spectral peak detection."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from detect_mexican_pacific_spectral_events import (
    SPECTRAL_BANDS,
    detect_spectral_peaks,
    validate_input,
)


def make_dataframe(signal=None):
    """Create a physically consistent synthetic spectral forecast."""

    leads = np.arange(0, 241, 6, dtype=float)

    if signal is None:
        signal = np.full(len(leads), 0.3)

    signal = np.asarray(signal, dtype=float)

    assert len(signal) == len(leads)

    rows = []

    for lead, height in zip(leads, signal):

        row = {
            "sector": "Test Coast",
            "lead_time_hours": lead,
            "swh_mean_m": 4.0,
            "swh_mean_h2_m2": 16.0,
            "spectral_sampling_supported": True,
            "n_spectral_valid_cells": 10,
        }

        for band in SPECTRAL_BANDS:

            value = (
                float(height)
                if band == "h1012"
                else 0.05
            )

            row[f"{band}_mean_m"] = value
            row[f"{band}_mean_h2_m2"] = value**2

        rows.append(row)

    return pd.DataFrame(rows)


def test_single_interior_peak():

    signal = np.full(41, 0.1)
    signal[10] = 1.0

    result = detect_spectral_peaks(
        make_dataframe(signal)
    )

    selected = result[
        result.band == "h1012"
    ]

    assert len(selected) == 1

    peak = selected.iloc[0]

    assert peak.peak_lead_h == 60.0
    assert peak.peak_band_hs_m == 1.0
    assert peak.peak_prominence_m == pytest.approx(0.9)
    assert peak.peak_energy_fraction == pytest.approx(
        1.0 / 16.0
    )


def test_flat_signal():

    result = detect_spectral_peaks(
        make_dataframe(
            np.full(41, 0.4)
        )
    )

    assert result.empty


def test_weak_signal():

    signal = np.full(41, 0.1)
    signal[10] = 0.25

    result = detect_spectral_peaks(
        make_dataframe(signal)
    )

    assert result.empty


def test_two_separated_peaks():

    signal = np.full(41, 0.1)

    signal[10] = 1.0
    signal[20] = 0.8

    result = detect_spectral_peaks(
        make_dataframe(signal)
    )

    selected = result[
        result.band == "h1012"
    ]

    assert selected.peak_lead_h.tolist() == [
        60.0,
        120.0,
    ]


def test_right_boundary_peak():

    signal = np.full(41, 0.1)
    signal[-1] = 0.8

    result = detect_spectral_peaks(
        make_dataframe(signal)
    )

    selected = result[
        result.band == "h1012"
    ]

    assert len(selected) == 1

    peak = selected.iloc[0]

    assert peak.peak_lead_h == 240.0
    assert bool(peak.right_boundary_peak)
    assert not bool(peak.left_boundary_peak)
    assert np.isnan(peak.peak_prominence_m)


def test_excess_spectral_energy():

    df = make_dataframe()

    df["h1012_mean_h2_m2"] = 20.0

    with pytest.raises(
        ValueError,
        match="spectral energy exceeds total",
    ):
        validate_input(df)


def test_irregular_cadence():

    df = make_dataframe()

    df.loc[
        df.lead_time_hours == 60,
        "lead_time_hours",
    ] = 61

    with pytest.raises(
        ValueError,
        match="irregular forecast cadence",
    ):
        validate_input(df)


def test_missing_required_column():

    df = make_dataframe().drop(
        columns=["h1417_mean_h2_m2"]
    )

    with pytest.raises(
        ValueError,
        match="Missing columns",
    ):
        validate_input(df)


def test_negative_spectral_height():

    df = make_dataframe()

    df.loc[
        df.lead_time_hours == 60,
        "h1012_mean_m",
    ] = -0.1

    with pytest.raises(
        ValueError,
        match="negative wave-height diagnostics",
    ):
        validate_input(df)


def test_boundary_diagnostics_right_monotonic():

    signal = np.full(41, 0.1)

    signal[-5:] = [
        0.20,
        0.35,
        0.50,
        0.65,
        0.80,
    ]

    result = detect_spectral_peaks(
        make_dataframe(signal)
    )

    peak = result.loc[
        (result.band == "h1012")
        & (result.right_boundary_peak)
    ].iloc[0]

    assert peak.peak_lead_h == 240.0

    assert peak.boundary_change_24h_m == pytest.approx(
        0.60
    )

    assert peak.boundary_relative_change_24h == pytest.approx(
        3.0
    )

    assert bool(peak.boundary_increase_ge_0p10m)

    assert bool(peak.boundary_monotonic_24h)


def test_boundary_diagnostics_left_monotonic():

    signal = np.full(41, 0.1)

    signal[:5] = [
        0.80,
        0.65,
        0.50,
        0.35,
        0.20,
    ]

    result = detect_spectral_peaks(
        make_dataframe(signal)
    )

    peak = result.loc[
        (result.band == "h1012")
        & (result.left_boundary_peak)
    ].iloc[0]

    assert peak.peak_lead_h == 0.0

    assert peak.boundary_change_24h_m == pytest.approx(
        0.60
    )

    assert peak.boundary_relative_change_24h == pytest.approx(
        3.0
    )

    assert bool(peak.boundary_increase_ge_0p10m)

    assert bool(peak.boundary_monotonic_24h)


def test_boundary_weak_change():

    signal = np.full(41, 0.1)

    signal[-5:] = [
        0.40,
        0.42,
        0.44,
        0.46,
        0.48,
    ]

    result = detect_spectral_peaks(
        make_dataframe(signal)
    )

    peak = result.loc[
        (result.band == "h1012")
        & (result.right_boundary_peak)
    ].iloc[0]

    assert peak.boundary_change_24h_m == pytest.approx(
        0.08
    )

    assert not bool(peak.boundary_increase_ge_0p10m)

    assert bool(peak.boundary_monotonic_24h)


def test_interior_peak_has_no_boundary_diagnostics():

    signal = np.full(41, 0.1)

    signal[10] = 1.0

    result = detect_spectral_peaks(
        make_dataframe(signal)
    )

    peak = result.loc[
        (result.band == "h1012")
        & (~result.left_boundary_peak)
        & (~result.right_boundary_peak)
    ].iloc[0]

    assert pd.isna(
        peak.boundary_change_24h_m
    )

    assert pd.isna(
        peak.boundary_relative_change_24h
    )

    assert pd.isna(
        peak.boundary_increase_ge_0p10m
    )

    assert pd.isna(
        peak.boundary_monotonic_24h
    )
