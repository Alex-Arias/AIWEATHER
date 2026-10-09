"""Tests for operational tropical-cyclone translation speed."""

import numpy as np
import pandas as pd
import pytest

from scripts.tc_translation_speed import calculate_translation_speed


def test_equatorial_translation():
    track = pd.DataFrame({
        "lead_time_hours": [0, 6, 12],
        "latitude": [0.0, 0.0, 0.0],
        "longitude": [0.0, 1.0, 2.0],
    })

    result = calculate_translation_speed(track)

    assert np.isnan(result.translation_speed_kmh.iloc[0])

    np.testing.assert_allclose(
        result.translation_speed_kmh.iloc[1:],
        [18.5325, 18.5325],
        rtol=0.001,
    )


def test_longitude_wraparound():
    track = pd.DataFrame({
        "lead_time_hours": [0, 6],
        "latitude": [0.0, 0.0],
        "longitude": [359.0, 0.0],
    })

    result = calculate_translation_speed(track)

    assert result.translation_speed_kmh.iloc[1] == pytest.approx(
        18.5325,
        rel=0.001,
    )


def test_irregular_time_intervals():
    track = pd.DataFrame({
        "lead_time_hours": [0, 3, 9],
        "latitude": [0.0, 0.0, 0.0],
        "longitude": [0.0, 1.0, 2.0],
    })

    result = calculate_translation_speed(track)

    assert result.translation_speed_kmh.iloc[1] == pytest.approx(
        37.065,
        rel=0.001,
    )

    assert result.translation_speed_kmh.iloc[2] == pytest.approx(
        18.5325,
        rel=0.001,
    )


def test_stationary_cyclone():
    track = pd.DataFrame({
        "lead_time_hours": [0, 6],
        "latitude": [15.0, 15.0],
        "longitude": [-105.0, -105.0],
    })

    result = calculate_translation_speed(track)

    assert result.translation_speed_kmh.iloc[1] == pytest.approx(0.0)


def test_nonincreasing_time_rejected():
    track = pd.DataFrame({
        "lead_time_hours": [0, 6, 6],
        "latitude": [15.0, 16.0, 17.0],
        "longitude": [-105.0, -106.0, -107.0],
    })

    with pytest.raises(ValueError, match="strictly increasing"):
        calculate_translation_speed(track)


def test_input_dataframe_unchanged():
    track = pd.DataFrame({
        "lead_time_hours": [0, 6],
        "latitude": [15.0, 16.0],
        "longitude": [-105.0, -106.0],
    })

    original = track.copy(deep=True)

    calculate_translation_speed(track)

    pd.testing.assert_frame_equal(track, original)
