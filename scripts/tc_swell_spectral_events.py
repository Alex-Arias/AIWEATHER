"""Experimental TC/coastal spectral-event integration.

Phase B2 associates existing AIFS2 spectral events and quality
records with accepted Phase B1 coastal-wave guidance.

Temporal association does not establish cyclone-generated swell
attribution.
"""

import numpy as np
import pandas as pd


EVENT_KEYS = [
    "sector",
    "band",
    "onset_lead_h",
    "end_lead_h",
    "primary_peak_lead_h",
]

TC_FIELDS = [
    "tc_latitude",
    "tc_longitude",
    "tc_pressure_hpa",
    "tc_wind_ms",
    "tc_r67_km",
    "tc_r90_km",
]

PEAK_FIELDS = [
    "valid_time",
    "tc_guidance_available",
    "tc_support_status",
    "tc_jointly_supported",
    *TC_FIELDS,
    "spectral_sampling_supported",
    "directional_qc_supported",
]

QC_EXTRA_FIELDS = [
    "record_type",
    "temporal_resolution",
    "boundary_change_24h_m",
    "boundary_monotonic_24h",
    "boundary_continuation_candidate",
]


def _require_columns(frame, columns, name):
    missing = set(columns) - set(frame.columns)
    if missing:
        raise ValueError(
            f"{name}: missing columns {sorted(missing)}"
        )


def _parse_boolean(series, name, allow_missing=False):
    """Parse boolean values without truthiness coercion."""
    mapping = {
        True: True,
        False: False,
        "True": True,
        "False": False,
        "true": True,
        "false": False,
    }

    values = series.map(mapping)

    invalid = values.isna() & (
        series.notna() | (not allow_missing)
    )

    if invalid.any():
        raise ValueError(f"{name}: invalid boolean values")

    return values.astype("boolean")


def validate_spectral_event_inputs(events, quality, b1):
    """Validate event/QC consistency and accepted B1 structure."""
    for name, frame in (
        ("events", events),
        ("quality", quality),
        ("b1", b1),
    ):
        if frame.empty:
            raise ValueError(f"{name}: empty input")

    _require_columns(
        events,
        EVENT_KEYS + [
            "duration_h",
            "primary_peak_band_hs_m",
            "n_constituent_peaks",
            "constituent_peak_leads_h",
            "left_censored",
            "right_censored",
            "ambiguous_overlap",
            "minimum_interpeak_valley_ratio",
        ],
        "events",
    )

    _require_columns(
        quality,
        list(events.columns) + QC_EXTRA_FIELDS,
        "quality",
    )

    _require_columns(
        b1,
        [
            "sector",
            "lead_time_hours",
            *PEAK_FIELDS,
        ],
        "b1",
    )

    if events.duplicated(EVENT_KEYS).any():
        raise ValueError("events: duplicate event keys")

    if quality.duplicated(EVENT_KEYS).any():
        raise ValueError("quality: duplicate event keys")

    if b1.duplicated(
        ["sector", "lead_time_hours"]
    ).any():
        raise ValueError("b1: duplicate sector/lead keys")

    for name, frame in (
        ("events", events),
        ("quality", quality),
    ):
        for column in (
            "onset_lead_h",
            "end_lead_h",
            "duration_h",
            "primary_peak_lead_h",
        ):
            values = pd.to_numeric(
                frame[column],
                errors="raise",
            )

            if not np.isfinite(
                values.to_numpy(dtype=float)
            ).all():
                raise ValueError(
                    f"{name}: invalid {column}"
                )

        onset = frame["onset_lead_h"]
        end = frame["end_lead_h"]
        peak = frame["primary_peak_lead_h"]
        duration = frame["duration_h"]

        if not (
            (onset >= 0)
            & (onset <= peak)
            & (peak <= end)
        ).all():
            raise ValueError(
                f"{name}: invalid event timing"
            )

        if not np.allclose(
            (end - onset).to_numpy(dtype=float),
            duration.to_numpy(dtype=float),
            rtol=0,
            atol=1e-9,
        ):
            raise ValueError(
                f"{name}: inconsistent event duration"
            )

        for column in (
            "left_censored",
            "right_censored",
            "ambiguous_overlap",
        ):
            _parse_boolean(
                frame[column],
                f"{name}.{column}",
            )

    common = sorted(
        set(events.columns)
        & set(quality.columns)
    )

    paired = events.merge(
        quality,
        on=EVENT_KEYS,
        how="outer",
        validate="one_to_one",
        indicator=True,
        suffixes=("_event", "_qc"),
    )

    if (
        len(paired) != len(events)
        or len(paired) != len(quality)
        or not paired["_merge"].eq("both").all()
    ):
        raise ValueError(
            "Event/QC keys do not match one-to-one"
        )

    for column in common:
        if column in EVENT_KEYS:
            continue

        left = paired[f"{column}_event"]
        right = paired[f"{column}_qc"]

        if (
            pd.api.types.is_numeric_dtype(left)
            and pd.api.types.is_numeric_dtype(right)
        ):
            equal = np.isclose(
                left.to_numpy(dtype=float),
                right.to_numpy(dtype=float),
                equal_nan=True,
                rtol=1e-12,
                atol=1e-12,
            )
        else:
            equal = (
                left.fillna("<MISSING>").astype(str)
                == right.fillna("<MISSING>").astype(str)
            ).to_numpy()

        if not equal.all():
            raise ValueError(
                f"Event/QC disagreement: {column}"
            )

    leads = pd.to_numeric(
        b1["lead_time_hours"],
        errors="raise",
    )

    if (
        not np.isfinite(
            leads.to_numpy(dtype=float)
        ).all()
        or (leads < 0).any()
    ):
        raise ValueError("b1: invalid forecast leads")

    available = _parse_boolean(
        b1["tc_guidance_available"],
        "b1.tc_guidance_available",
    )

    supported = _parse_boolean(
        b1["tc_jointly_supported"],
        "b1.tc_jointly_supported",
        allow_missing=True,
    )

    statuses = b1["tc_support_status"]

    allowed = {
        "supported",
        "limited",
        "insufficient",
        "unavailable",
    }

    if statuses.isna().any() or not statuses.isin(
        allowed
    ).all():
        raise ValueError(
            "b1: invalid TC support status"
        )

    inconsistent = (
        (available & supported.isna())
        | (~available & supported.notna())
        | (available & statuses.eq("unavailable"))
        | (~available & statuses.ne("unavailable"))
        | (supported.fillna(False)
           & statuses.ne("supported"))
        | (available
           & ~supported.fillna(False)
           & statuses.eq("supported"))
    )

    if inconsistent.any():
        raise ValueError(
            "b1: contradictory TC availability/support"
        )

    for column in TC_FIELDS:
        if b1.loc[~available, column].notna().any():
            raise ValueError(
                f"b1: unavailable TC field populated: {column}"
            )

    tc_leads = leads[available]

    if tc_leads.empty:
        raise ValueError(
            "b1: no available TC guidance"
        )

    tc_horizon = float(tc_leads.max())

    if (
        (available & (leads > tc_horizon)).any()
        or (
            ~available
            & (leads <= tc_horizon)
        ).any()
    ):
        raise ValueError(
            "b1: inconsistent TC guidance horizon"
        )

    return tc_horizon


def build_spectral_event_guidance(events, quality, b1):
    """Build one experimental record per spectral event."""
    horizon = validate_spectral_event_inputs(
        events, quality, b1
    )

    result = events.copy()

    qc = quality[
        EVENT_KEYS + QC_EXTRA_FIELDS
    ].copy()

    result = result.merge(
        qc,
        on=EVENT_KEYS,
        how="left",
        validate="one_to_one",
        sort=False,
    )

    if len(result) != len(events):
        raise ValueError(
            "Event count changed during QC integration"
        )

    peak = b1[
        [
            "sector",
            "lead_time_hours",
            *PEAK_FIELDS,
        ]
    ].rename(
        columns={
            "lead_time_hours": "primary_peak_lead_h",
            "valid_time": "peak_valid_time",
            "tc_guidance_available": "peak_tc_guidance_available",
            "tc_support_status": "peak_tc_support_status",
            "tc_jointly_supported": "peak_tc_jointly_supported",
            "spectral_sampling_supported": "peak_spectral_sampling_supported",
            "directional_qc_supported": "peak_directional_qc_supported",
            **{
                field: f"peak_{field}"
                for field in TC_FIELDS
            },
        }
    )

    result = result.merge(
        peak,
        on=["sector", "primary_peak_lead_h"],
        how="left",
        validate="many_to_one",
        indicator=True,
        sort=False,
    )

    if not result["_merge"].eq("both").all():
        raise ValueError(
            "Spectral event peak missing from B1 grid"
        )

    result = result.drop(columns="_merge")

    onset = result["onset_lead_h"]
    end = result["end_lead_h"]

    within = end <= horizon
    beyond = onset > horizon
    crossing = ~(within | beyond)

    result["event_tc_horizon_class"] = np.select(
        [within, crossing, beyond],
        ["within", "crossing", "beyond"],
        default="unknown",
    )

    result["peak_within_tc_horizon"] = (
        result["primary_peak_lead_h"] <= horizon
    )

    available = _parse_boolean(
        result["peak_tc_guidance_available"],
        "peak_tc_guidance_available",
    )

    if not available.eq(
        result["peak_within_tc_horizon"]
    ).all():
        raise ValueError(
            "Peak TC availability inconsistent with horizon"
        )

    result["tc_guidance_horizon_h"] = horizon

    if len(result) != len(events):
        raise ValueError(
            "Spectral event record count changed"
        )

    return result
