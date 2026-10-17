"""Regression tests for spectral-event quality classification."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(
    0,
    str(ROOT / "scripts"),
)

from classify_mexican_pacific_spectral_events import (
    QUALITY_COLUMNS,
    classify_spectral_events,
)


def make_events(
    duration=24.0,
    left_censored=False,
    right_censored=False,
    ambiguous_overlap=False,
):
    return pd.DataFrame([
        {
            "sector": "Test Coast",
            "band": "h1417",
            "onset_lead_h": 48.0,
            "end_lead_h": 48.0 + duration,
            "duration_h": duration,
            "primary_peak_lead_h": 48.0,
            "primary_peak_band_hs_m": 0.8,
            "n_constituent_peaks": 1,
            "constituent_peak_leads_h": "48",
            "left_censored": left_censored,
            "right_censored": right_censored,
            "ambiguous_overlap": ambiguous_overlap,
            "minimum_interpeak_valley_ratio": np.nan,
        }
    ])


def make_peaks(
    change=0.20,
    monotonic=True,
    right_boundary=True,
):
    return pd.DataFrame([
        {
            "sector": "Test Coast",
            "band": "h1417",
            "peak_lead_h": 240.0,
            "peak_band_hs_m": 0.7,
            "left_boundary_peak": False,
            "right_boundary_peak": right_boundary,
            "boundary_change_24h_m": change,
            "boundary_monotonic_24h": monotonic,
        }
    ])


def empty_events():
    return make_events().iloc[0:0].copy()


def empty_peaks():
    return make_peaks().iloc[0:0].copy()


def test_resolved_episode():

    result = classify_spectral_events(
        make_events(duration=24),
        empty_peaks(),
    )

    assert len(result) == 1

    assert (
        result.iloc[0].temporal_resolution
        == "resolved"
    )

    assert (
        result.iloc[0].record_type
        == "segmented_episode"
    )


def test_single_sample_episode():

    result = classify_spectral_events(
        make_events(duration=0),
        empty_peaks(),
    )

    assert len(result) == 1

    assert (
        result.iloc[0].temporal_resolution
        == "single_sample"
    )


def test_short_episode():

    result = classify_spectral_events(
        make_events(duration=6),
        empty_peaks(),
    )

    assert (
        result.iloc[0].temporal_resolution
        == "short"
    )


def test_strong_boundary_candidate():

    result = classify_spectral_events(
        empty_events(),
        make_peaks(
            change=0.20,
            monotonic=True,
        ),
    )

    assert len(result) == 1

    row = result.iloc[0]

    assert (
        row.record_type
        == "boundary_continuation_candidate"
    )

    assert row.temporal_resolution == "right_boundary_only"

    assert bool(row.right_censored)

    assert bool(row.boundary_continuation_candidate)

    assert np.isnan(row.onset_lead_h)


def test_weak_boundary_excluded():

    result = classify_spectral_events(
        empty_events(),
        make_peaks(
            change=0.05,
            monotonic=True,
        ),
    )

    assert result.empty

    assert list(result.columns) == QUALITY_COLUMNS


def test_flags_preserved():

    result = classify_spectral_events(
        make_events(
            duration=24,
            left_censored=True,
            ambiguous_overlap=True,
        ),
        empty_peaks(),
    )

    row = result.iloc[0]

    assert bool(row.left_censored)

    assert bool(row.ambiguous_overlap)


def test_invalid_thresholds():

    with pytest.raises(ValueError):

        classify_spectral_events(
            make_events(),
            empty_peaks(),
            minimum_resolved_duration_h=0,
        )

    with pytest.raises(ValueError):

        classify_spectral_events(
            make_events(),
            empty_peaks(),
            minimum_boundary_change_m=-0.1,
        )


def test_empty_catalogs():

    result = classify_spectral_events(
        empty_events(),
        empty_peaks(),
    )

    assert result.empty

    assert list(result.columns) == QUALITY_COLUMNS


def test_overlapping_record_types_preserved():

    events = make_events(duration=192)

    events.loc[
        0,
        "right_censored",
    ] = True

    peaks = make_peaks(
        change=0.20,
        monotonic=True,
    )

    result = classify_spectral_events(
        events,
        peaks,
    )

    assert len(result) == 2

    assert set(result.record_type) == {
        "segmented_episode",
        "boundary_continuation_candidate",
    }


def test_nonmonotonic_boundary_excluded():

    result = classify_spectral_events(
        empty_events(),
        make_peaks(
            change=0.20,
            monotonic=False,
        ),
    )

    assert result.empty


def test_left_boundary_not_continuation():

    peaks = make_peaks(
        change=0.20,
        monotonic=True,
        right_boundary=False,
    )

    peaks.loc[
        0,
        "left_boundary_peak",
    ] = True

    result = classify_spectral_events(
        empty_events(),
        peaks,
    )

    assert result.empty


def test_input_frames_not_modified():

    events = make_events()
    peaks = make_peaks()

    original_events = events.copy(deep=True)
    original_peaks = peaks.copy(deep=True)

    classify_spectral_events(
        events,
        peaks,
    )

    pd.testing.assert_frame_equal(
        events,
        original_events,
    )

    pd.testing.assert_frame_equal(
        peaks,
        original_peaks,
    )
