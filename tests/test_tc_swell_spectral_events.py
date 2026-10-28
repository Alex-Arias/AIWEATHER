"""Regression tests for experimental B2 spectral-event integration."""

import pandas as pd
import pytest

from scripts.tc_swell_spectral_events import (
    build_spectral_event_guidance,
    validate_spectral_event_inputs,
)


def sample_inputs(horizon=12):
    """Construct synthetic event, QC, and B1 datasets."""
    events = pd.DataFrame([
        {
            "sector": "Test Coast",
            "band": "h1012",
            "onset_lead_h": 0,
            "end_lead_h": 12,
            "duration_h": 12,
            "primary_peak_lead_h": 6,
            "primary_peak_band_hs_m": 1.5,
            "n_constituent_peaks": 1,
            "constituent_peak_leads_h": "6",
            "left_censored": False,
            "right_censored": False,
            "ambiguous_overlap": False,
            "minimum_interpeak_valley_ratio": float("nan"),
        },
        {
            "sector": "Test Coast",
            "band": "h1214",
            "onset_lead_h": 6,
            "end_lead_h": 24,
            "duration_h": 18,
            "primary_peak_lead_h": 18,
            "primary_peak_band_hs_m": 2.0,
            "n_constituent_peaks": 1,
            "constituent_peak_leads_h": "18",
            "left_censored": False,
            "right_censored": False,
            "ambiguous_overlap": True,
            "minimum_interpeak_valley_ratio": float("nan"),
        },
        {
            "sector": "Test Coast",
            "band": "h1417",
            "onset_lead_h": 18,
            "end_lead_h": 24,
            "duration_h": 6,
            "primary_peak_lead_h": 24,
            "primary_peak_band_hs_m": 0.8,
            "n_constituent_peaks": 1,
            "constituent_peak_leads_h": "24",
            "left_censored": False,
            "right_censored": True,
            "ambiguous_overlap": False,
            "minimum_interpeak_valley_ratio": float("nan"),
        },
    ])

    quality = events.copy()

    quality["record_type"] = "segmented_episode"
    quality["temporal_resolution"] = "resolved"
    quality["boundary_change_24h_m"] = float("nan")
    quality["boundary_monotonic_24h"] = False
    quality["boundary_continuation_candidate"] = False

    rows = []

    for lead in range(0, 25, 6):
        available = lead <= horizon

        rows.append({
            "sector": "Test Coast",
            "lead_time_hours": lead,
            "valid_time": (
                pd.Timestamp("2026-10-05 12:00")
                + pd.Timedelta(hours=lead)
            ).isoformat(),
            "tc_guidance_available": available,
            "tc_support_status": (
                "supported" if available else "unavailable"
            ),
            "tc_jointly_supported": (
                True if available else pd.NA
            ),
            "tc_latitude": 15.0 if available else float("nan"),
            "tc_longitude": -110.0 if available else float("nan"),
            "tc_pressure_hpa": 990.0 if available else float("nan"),
            "tc_wind_ms": 25.0 if available else float("nan"),
            "tc_r67_km": 40.0 if available else float("nan"),
            "tc_r90_km": 60.0 if available else float("nan"),
            "spectral_sampling_supported": True,
            "directional_qc_supported": True,
        })

    b1 = pd.DataFrame(rows)

    return events, quality, b1


def test_valid_integration():
    events, quality, b1 = sample_inputs()

    result = build_spectral_event_guidance(
        events, quality, b1
    )

    assert len(result) == 3
    assert result["event_tc_horizon_class"].tolist() == [
        "within", "crossing", "beyond"
    ]


def test_peak_support_classification():
    events, quality, b1 = sample_inputs()

    result = build_spectral_event_guidance(
        events, quality, b1
    )

    assert result["peak_tc_support_status"].tolist() == [
        "supported", "unavailable", "unavailable"
    ]


def test_no_tc_extrapolation():
    events, quality, b1 = sample_inputs()

    result = build_spectral_event_guidance(
        events, quality, b1
    )

    unavailable = ~result[
        "peak_tc_guidance_available"
    ].astype(bool)

    assert result.loc[
        unavailable, "peak_tc_latitude"
    ].isna().all()


def test_original_event_fields_preserved():
    events, quality, b1 = sample_inputs()

    result = build_spectral_event_guidance(
        events, quality, b1
    )

    for column in events.columns:
        pd.testing.assert_series_equal(
            result[column],
            events[column],
        )


def test_qc_fields_preserved():
    events, quality, b1 = sample_inputs()

    result = build_spectral_event_guidance(
        events, quality, b1
    )

    for column in (
        "record_type",
        "temporal_resolution",
        "boundary_change_24h_m",
        "boundary_monotonic_24h",
        "boundary_continuation_candidate",
    ):
        assert column in result.columns


def test_duplicate_event_keys_rejected():
    events, quality, b1 = sample_inputs()

    events = pd.concat(
        [events, events.iloc[[0]]],
        ignore_index=True,
    )

    with pytest.raises(ValueError, match="duplicate"):
        validate_spectral_event_inputs(
            events, quality, b1
        )


def test_duplicate_qc_keys_rejected():
    events, quality, b1 = sample_inputs()

    quality = pd.concat(
        [quality, quality.iloc[[0]]],
        ignore_index=True,
    )

    with pytest.raises(ValueError, match="duplicate"):
        validate_spectral_event_inputs(
            events, quality, b1
        )


def test_duplicate_b1_keys_rejected():
    events, quality, b1 = sample_inputs()

    b1 = pd.concat(
        [b1, b1.iloc[[0]]],
        ignore_index=True,
    )

    with pytest.raises(ValueError, match="duplicate"):
        validate_spectral_event_inputs(
            events, quality, b1
        )


def test_invalid_duration_rejected():
    events, quality, b1 = sample_inputs()

    events.loc[0, "duration_h"] = 99

    with pytest.raises(ValueError, match="duration"):
        validate_spectral_event_inputs(
            events, quality, b1
        )


def test_peak_outside_event_rejected():
    events, quality, b1 = sample_inputs()

    events.loc[0, "primary_peak_lead_h"] = 18

    with pytest.raises(ValueError, match="timing"):
        validate_spectral_event_inputs(
            events, quality, b1
        )


def test_event_qc_disagreement_rejected():
    events, quality, b1 = sample_inputs()

    quality.loc[0, "primary_peak_band_hs_m"] = 9.0

    with pytest.raises(ValueError, match="disagreement"):
        validate_spectral_event_inputs(
            events, quality, b1
        )


def test_missing_peak_record_rejected():
    events, quality, b1 = sample_inputs()

    b1 = b1.loc[
        b1["lead_time_hours"] != 18
    ].copy()

    with pytest.raises(ValueError, match="missing from B1"):
        build_spectral_event_guidance(
            events, quality, b1
        )


def test_invalid_tc_support_rejected():
    events, quality, b1 = sample_inputs()

    b1.loc[0, "tc_support_status"] = "unknown"

    with pytest.raises(ValueError, match="support status"):
        validate_spectral_event_inputs(
            events, quality, b1
        )


def test_contradictory_tc_support_rejected():
    events, quality, b1 = sample_inputs()

    b1.loc[0, "tc_jointly_supported"] = False

    with pytest.raises(ValueError, match="contradictory"):
        validate_spectral_event_inputs(
            events, quality, b1
        )


def test_unavailable_tc_fields_rejected():
    events, quality, b1 = sample_inputs()

    b1.loc[
        b1["lead_time_hours"] == 18,
        "tc_latitude",
    ] = 15.0

    with pytest.raises(ValueError, match="unavailable TC field"):
        validate_spectral_event_inputs(
            events, quality, b1
        )


def test_invalid_boolean_rejected():
    events, quality, b1 = sample_inputs()

    events["ambiguous_overlap"] = "maybe"

    with pytest.raises(ValueError, match="boolean"):
        validate_spectral_event_inputs(
            events, quality, b1
        )


def test_dynamic_tc_horizon():
    events, quality, b1 = sample_inputs(horizon=18)

    result = build_spectral_event_guidance(
        events, quality, b1
    )

    assert result["tc_guidance_horizon_h"].eq(18).all()

    assert result["event_tc_horizon_class"].tolist() == [
        "within", "crossing", "crossing"
    ]


def test_missing_required_column_rejected():
    events, quality, b1 = sample_inputs()

    events = events.drop(columns="duration_h")

    with pytest.raises(ValueError, match="missing columns"):
        validate_spectral_event_inputs(
            events, quality, b1
        )


def test_event_qc_key_mismatch_rejected():
    events, quality, b1 = sample_inputs()

    quality.loc[0, "sector"] = "Different Coast"

    with pytest.raises(ValueError, match="one-to-one"):
        validate_spectral_event_inputs(
            events, quality, b1
        )
