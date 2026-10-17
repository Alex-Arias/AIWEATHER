"""Regression tests for coastal spectral-event segmentation."""

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
)

from segment_mexican_pacific_spectral_events import (
    segment_spectral_events,
)


def make_guidance(signal):
    """Build a physically consistent synthetic spectral forecast."""

    signal = np.asarray(signal, dtype=float)

    leads = np.arange(
        len(signal),
        dtype=float,
    ) * 6.0

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


def run_segmentation(signal, **kwargs):

    guidance = make_guidance(signal)

    peaks = detect_spectral_peaks(
        guidance,
        min_height_m=0.30,
        min_prominence_m=0.10,
        min_separation_h=12.0,
    )

    events = segment_spectral_events(
        guidance,
        peaks,
        **kwargs,
    )

    return peaks, events


def test_single_isolated_episode():

    signal = np.full(41, 0.10)

    signal[8:13] = [
        0.30,
        0.60,
        1.00,
        0.60,
        0.30,
    ]

    peaks, events = run_segmentation(signal)

    assert len(events) == 1

    event = events.iloc[0]

    assert event.primary_peak_lead_h == 60.0
    assert event.primary_peak_band_hs_m == pytest.approx(1.0)
    assert event.n_constituent_peaks == 1
    assert event.onset_lead_h == 54.0
    assert event.end_lead_h == 66.0
    assert event.duration_h == 12.0
    assert not bool(event.left_censored)
    assert not bool(event.right_censored)


def test_shallow_valley_merges_peaks():

    signal = np.full(41, 0.10)

    signal[8:19] = [
        0.40,
        0.80,
        1.20,
        1.10,
        1.00,
        1.05,
        1.15,
        1.25,
        1.10,
        0.80,
        0.40,
    ]

    peaks, events = run_segmentation(signal)

    interior = peaks.loc[
        ~(
            peaks.left_boundary_peak
            | peaks.right_boundary_peak
        )
    ]

    assert len(interior) == 2
    assert len(events) == 1

    event = events.iloc[0]

    assert event.n_constituent_peaks == 2
    assert event.constituent_peak_leads_h == "60;90"
    assert event.primary_peak_lead_h == 90.0
    assert event.minimum_interpeak_valley_ratio >= 0.80
    assert not bool(event.ambiguous_overlap)


def test_separated_peaks_remain_distinct():

    signal = np.full(41, 0.10)

    signal[8:13] = [
        0.30,
        0.60,
        1.00,
        0.60,
        0.30,
    ]

    signal[20:25] = [
        0.30,
        0.60,
        0.90,
        0.60,
        0.30,
    ]

    _, events = run_segmentation(signal)

    assert len(events) == 2

    assert events.n_constituent_peaks.tolist() == [
        1,
        1,
    ]

    assert sorted(
        events.primary_peak_lead_h.tolist()
    ) == [
        60.0,
        132.0,
    ]


def test_left_censored_episode():

    signal = np.full(41, 0.10)

    signal[:8] = [
        0.70,
        0.80,
        1.00,
        0.90,
        0.70,
        0.50,
        0.30,
        0.10,
    ]

    _, events = run_segmentation(signal)

    assert len(events) == 1

    event = events.iloc[0]

    assert event.primary_peak_lead_h == 12.0
    assert event.onset_lead_h == 0.0
    assert bool(event.left_censored)
    assert not bool(event.right_censored)


def test_right_censored_episode():

    signal = np.full(41, 0.10)

    signal[33:] = [
        0.10,
        0.30,
        0.50,
        0.80,
        1.00,
        0.90,
        0.80,
        0.70,
    ]

    _, events = run_segmentation(signal)

    assert len(events) == 1

    event = events.iloc[0]

    assert event.primary_peak_lead_h == 222.0
    assert event.end_lead_h == 240.0
    assert bool(event.right_censored)
    assert not bool(event.left_censored)


def test_boundary_only_peak_excluded():

    signal = np.full(41, 0.10)

    signal[-5:] = [
        0.20,
        0.35,
        0.50,
        0.65,
        0.80,
    ]

    peaks, events = run_segmentation(signal)

    assert len(peaks) == 1
    assert bool(peaks.iloc[0].right_boundary_peak)

    assert events.empty


def test_invalid_peak_height_rejected():

    signal = np.full(41, 0.10)
    signal[10] = 1.0

    guidance = make_guidance(signal)

    peaks = detect_spectral_peaks(guidance)

    peaks.loc[
        peaks.band == "h1012",
        "peak_band_hs_m",
    ] = 1.5

    with pytest.raises(
        ValueError,
        match="peak height disagrees",
    ):
        segment_spectral_events(
            guidance,
            peaks,
        )


def test_invalid_merge_ratio_rejected():

    signal = np.full(41, 0.10)
    signal[10] = 1.0

    guidance = make_guidance(signal)
    peaks = detect_spectral_peaks(guidance)

    with pytest.raises(
        ValueError,
        match="merge_valley_ratio",
    ):
        segment_spectral_events(
            guidance,
            peaks,
            merge_valley_ratio=1.5,
        )
