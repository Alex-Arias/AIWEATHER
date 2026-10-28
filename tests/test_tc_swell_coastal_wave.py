"""Regression tests for experimental TC/coastal-wave integration."""

import numpy as np
import pandas as pd
import pytest

from scripts.build_tc_swell_advisory import (
    build_coastal_wave_guidance,
)


INIT = pd.Timestamp("2026-10-05 12:00:00")


def timestamp(lead):
    return (
        INIT + pd.Timedelta(hours=lead)
    ).strftime("%Y-%m-%d %H:%M:%S")


@pytest.fixture
def guidance():
    wave_rows = []

    for sector in ("BC North", "Los Cabos"):
        for lead in (0, 6, 12):
            wave_rows.append({
                "sector": sector,
                "valid_time": timestamp(lead),
                "lead_time_hours": lead,
                "wave_height_m": 1.0 + lead / 12,
                "wave_period_s": 14.0,
            })

    wave = pd.DataFrame(wave_rows)

    tc = pd.DataFrame([
        {
            "valid_time": timestamp(0),
            "lead_time_hours": 0,
            "tc_jointly_supported": True,
            "tc_support_status": "supported",
            "consensus_latitude": 20.0,
            "consensus_longitude": -120.0,
            "pressure_mean_hpa": 990.0,
            "wind_mean_ms": 25.0,
            "r67_km": 80.0,
            "r90_km": 120.0,
        },
        {
            "valid_time": timestamp(6),
            "lead_time_hours": 6,
            "tc_jointly_supported": False,
            "tc_support_status": "limited",
            "consensus_latitude": 20.5,
            "consensus_longitude": -119.5,
            "pressure_mean_hpa": 995.0,
            "wind_mean_ms": 22.0,
            "r67_km": 90.0,
            "r90_km": 140.0,
        },
    ])

    return wave, tc


def test_valid_integration(guidance):
    wave, tc = guidance
    result = build_coastal_wave_guidance(wave, tc)

    assert len(result) == 6
    assert result["tc_guidance_available"].sum() == 4
    assert len(result.columns) == len(wave.columns) + 9


def test_original_wave_values_preserved(guidance):
    wave, tc = guidance
    result = build_coastal_wave_guidance(wave, tc)

    expected = wave.copy()
    expected["valid_time"] = pd.to_datetime(
        expected["valid_time"]
    )

    pd.testing.assert_frame_equal(
        result[expected.columns].reset_index(drop=True),
        expected.reset_index(drop=True),
        check_dtype=False,
    )


def test_three_state_support(guidance):
    wave, tc = guidance
    result = build_coastal_wave_guidance(wave, tc)

    support = result["tc_jointly_supported"]

    assert str(support.dtype) == "boolean"
    assert int((support == True).sum()) == 2
    assert int((support == False).sum()) == 2
    assert int(support.isna().sum()) == 2


def test_no_tc_extrapolation(guidance):
    wave, tc = guidance
    result = build_coastal_wave_guidance(wave, tc)

    beyond = result[result["lead_time_hours"] == 12]

    assert len(beyond) == 2
    assert not beyond["tc_guidance_available"].any()
    assert beyond["tc_latitude"].isna().all()
    assert beyond["tc_longitude"].isna().all()
    assert beyond["tc_wind_ms"].isna().all()
    assert beyond["tc_jointly_supported"].isna().all()
    assert (beyond["tc_support_status"] == "unavailable").all()


@pytest.mark.parametrize("dataset", ["wave", "tc"])
def test_duplicate_keys_rejected(guidance, dataset):
    wave, tc = guidance

    if dataset == "wave":
        wave = pd.concat([wave, wave.iloc[[0]]])
    else:
        tc = pd.concat([tc, tc.iloc[[0]]])

    with pytest.raises(ValueError, match="Duplicate"):
        build_coastal_wave_guidance(wave, tc)


@pytest.mark.parametrize("dataset", ["wave", "tc"])
def test_missing_columns_rejected(guidance, dataset):
    wave, tc = guidance

    if dataset == "wave":
        wave = wave.drop(columns=["sector"])
    else:
        tc = tc.drop(columns=["wind_mean_ms"])

    with pytest.raises(ValueError, match="Missing"):
        build_coastal_wave_guidance(wave, tc)


@pytest.mark.parametrize("dataset", ["wave", "tc"])
def test_inconsistent_initialization_rejected(guidance, dataset):
    wave, tc = guidance

    if dataset == "wave":
        wave.loc[0, "valid_time"] = timestamp(1)
    else:
        tc.loc[0, "valid_time"] = timestamp(1)

    with pytest.raises(
        ValueError,
        match="inconsistent forecast initialization",
    ):
        build_coastal_wave_guidance(wave, tc)


def test_cross_cycle_initialization_rejected(guidance):
    wave, tc = guidance

    tc["valid_time"] = (
        pd.to_datetime(tc["valid_time"])
        + pd.Timedelta(hours=6)
    ).dt.strftime("%Y-%m-%d %H:%M:%S")

    with pytest.raises(
        ValueError,
        match="initializations differ",
    ):
        build_coastal_wave_guidance(wave, tc)


@pytest.mark.parametrize(
    "invalid_lead",
    [-6, np.inf, np.nan, "INVALID"],
)
def test_invalid_wave_leads_rejected(guidance, invalid_lead):
    wave, tc = guidance

    wave["lead_time_hours"] = (
        wave["lead_time_hours"].astype(object)
    )
    wave.loc[0, "lead_time_hours"] = invalid_lead

    with pytest.raises((ValueError, TypeError, OverflowError)):
        build_coastal_wave_guidance(wave, tc)


@pytest.mark.parametrize(
    "invalid_lead",
    [-6, np.inf, np.nan, "INVALID"],
)
def test_invalid_tc_leads_rejected(guidance, invalid_lead):
    wave, tc = guidance

    tc["lead_time_hours"] = (
        tc["lead_time_hours"].astype(object)
    )
    tc.loc[0, "lead_time_hours"] = invalid_lead

    with pytest.raises((ValueError, TypeError, OverflowError)):
        build_coastal_wave_guidance(wave, tc)


@pytest.mark.parametrize(
    "invalid_support",
    [None, "INVALID"],
)
def test_invalid_tc_support_rejected(guidance, invalid_support):
    wave, tc = guidance

    tc["tc_jointly_supported"] = (
        tc["tc_jointly_supported"].astype(object)
    )
    tc.loc[0, "tc_jointly_supported"] = invalid_support

    with pytest.raises(ValueError):
        build_coastal_wave_guidance(wave, tc)


@pytest.mark.parametrize(
    "invalid_status",
    [None, "INVALID", "unavailable"],
)
def test_invalid_tc_status_rejected(guidance, invalid_status):
    wave, tc = guidance

    tc.loc[0, "tc_support_status"] = invalid_status

    with pytest.raises(ValueError):
        build_coastal_wave_guidance(wave, tc)


@pytest.mark.parametrize(
    "support,status",
    [
        (True, "limited"),
        (True, "insufficient"),
        (False, "supported"),
    ],
)
def test_contradictory_support_rejected(
    guidance,
    support,
    status,
):
    wave, tc = guidance

    tc.loc[0, "tc_jointly_supported"] = support
    tc.loc[0, "tc_support_status"] = status

    with pytest.raises(ValueError, match="disagree"):
        build_coastal_wave_guidance(wave, tc)
