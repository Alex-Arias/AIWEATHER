#!/usr/bin/env python3
"""Classify experimental Mexican Pacific spectral-energy detections.

The resulting records describe detection quality, not confirmed
swell arrivals, tropical-cyclone attribution, or coastal hazards.

Inputs
------
1. Segmented interior-peak event catalog.
2. Original spectral peak catalog.

The input catalogs are never modified.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


QUALITY_COLUMNS = [
    "record_type",
    "sector",
    "band",
    "onset_lead_h",
    "end_lead_h",
    "duration_h",
    "primary_peak_lead_h",
    "primary_peak_band_hs_m",
    "n_constituent_peaks",
    "constituent_peak_leads_h",
    "temporal_resolution",
    "left_censored",
    "right_censored",
    "ambiguous_overlap",
    "minimum_interpeak_valley_ratio",
    "boundary_change_24h_m",
    "boundary_monotonic_24h",
    "boundary_continuation_candidate",
]


def _boolean(series):
    """Normalize boolean columns loaded from CSV."""

    return (
        series.astype("string")
        .str.lower()
        .eq("true")
        .fillna(False)
        .to_numpy(dtype=bool)
    )


def classify_spectral_events(
    events: pd.DataFrame,
    peaks: pd.DataFrame,
    minimum_resolved_duration_h: float = 12.0,
    minimum_boundary_change_m: float = 0.10,
) -> pd.DataFrame:
    """Build a non-destructive spectral-event quality catalog."""

    if (
        not np.isfinite(minimum_resolved_duration_h)
        or minimum_resolved_duration_h <= 0
    ):
        raise ValueError(
            "minimum_resolved_duration_h must be positive"
        )

    if (
        not np.isfinite(minimum_boundary_change_m)
        or minimum_boundary_change_m <= 0
    ):
        raise ValueError(
            "minimum_boundary_change_m must be positive"
        )

    required_events = {
        "sector",
        "band",
        "onset_lead_h",
        "end_lead_h",
        "duration_h",
        "primary_peak_lead_h",
        "primary_peak_band_hs_m",
        "n_constituent_peaks",
        "constituent_peak_leads_h",
        "left_censored",
        "right_censored",
        "ambiguous_overlap",
        "minimum_interpeak_valley_ratio",
    }

    required_peaks = {
        "sector",
        "band",
        "peak_lead_h",
        "peak_band_hs_m",
        "left_boundary_peak",
        "right_boundary_peak",
        "boundary_change_24h_m",
        "boundary_monotonic_24h",
    }

    missing_events = required_events - set(events.columns)
    missing_peaks = required_peaks - set(peaks.columns)

    if missing_events:
        raise ValueError(
            f"Missing event columns: {sorted(missing_events)}"
        )

    if missing_peaks:
        raise ValueError(
            f"Missing peak columns: {sorted(missing_peaks)}"
        )

    episodes = events.copy()

    if not episodes.empty:

        durations = pd.to_numeric(
            episodes["duration_h"],
            errors="raise",
        ).to_numpy(dtype=float)

        if (
            not np.isfinite(durations).all()
            or (durations < 0).any()
        ):
            raise ValueError(
                "Event durations must be finite and nonnegative"
            )

        episodes["record_type"] = "segmented_episode"

        episodes["temporal_resolution"] = np.select(
            [
                durations == 0,
                (durations > 0)
                & (durations < minimum_resolved_duration_h),
                durations >= minimum_resolved_duration_h,
            ],
            [
                "single_sample",
                "short",
                "resolved",
            ],
            default="invalid",
        )

        episodes["boundary_change_24h_m"] = np.nan
        episodes["boundary_monotonic_24h"] = False
        episodes["boundary_continuation_candidate"] = False

    right = peaks.loc[
        _boolean(peaks["right_boundary_peak"])
    ].copy()

    changes = pd.to_numeric(
        right["boundary_change_24h_m"],
        errors="coerce",
    )

    strong = right.loc[
        (changes >= minimum_boundary_change_m)
        & _boolean(right["boundary_monotonic_24h"])
    ].copy()

    boundary_records = []

    for peak in strong.itertuples(index=False):

        boundary_records.append({
            "record_type": "boundary_continuation_candidate",
            "sector": peak.sector,
            "band": peak.band,
            "onset_lead_h": np.nan,
            "end_lead_h": float(peak.peak_lead_h),
            "duration_h": np.nan,
            "primary_peak_lead_h": float(peak.peak_lead_h),
            "primary_peak_band_hs_m": float(
                peak.peak_band_hs_m
            ),
            "n_constituent_peaks": 0,
            "constituent_peak_leads_h": "",
            "temporal_resolution": "right_boundary_only",
            "left_censored": False,
            "right_censored": True,
            "ambiguous_overlap": False,
            "minimum_interpeak_valley_ratio": np.nan,
            "boundary_change_24h_m": float(
                peak.boundary_change_24h_m
            ),
            "boundary_monotonic_24h": True,
            "boundary_continuation_candidate": True,
        })

    boundary = pd.DataFrame(
        boundary_records,
        columns=QUALITY_COLUMNS,
    )

    frames = []

    if not episodes.empty:
        frames.append(
            episodes.reindex(columns=QUALITY_COLUMNS)
        )

    if not boundary.empty:
        frames.append(
            boundary.reindex(columns=QUALITY_COLUMNS)
        )

    if frames:
        combined = pd.concat(
            frames,
            ignore_index=True,
        )
    else:
        combined = pd.DataFrame(
            columns=QUALITY_COLUMNS
        )

    if combined.empty:
        return pd.DataFrame(columns=QUALITY_COLUMNS)

    combined = combined.sort_values(
        [
            "sector",
            "band",
            "primary_peak_lead_h",
            "record_type",
        ],
        kind="stable",
    ).reset_index(drop=True)

    return combined[QUALITY_COLUMNS]


def main():

    parser = argparse.ArgumentParser(
        description=(
            "Classify experimental coastal spectral-energy "
            "episodes and boundary continuation candidates."
        )
    )

    parser.add_argument(
        "--events",
        required=True,
        type=Path,
    )

    parser.add_argument(
        "--peaks",
        required=True,
        type=Path,
    )

    parser.add_argument(
        "--output",
        required=True,
        type=Path,
    )

    parser.add_argument(
        "--minimum-resolved-duration-h",
        type=float,
        default=12.0,
    )

    parser.add_argument(
        "--minimum-boundary-change-m",
        type=float,
        default=0.10,
    )

    args = parser.parse_args()

    if args.output.exists():
        parser.error(
            f"Output already exists: {args.output}"
        )

    events = pd.read_csv(args.events)
    peaks = pd.read_csv(args.peaks)

    result = classify_spectral_events(
        events,
        peaks,
        minimum_resolved_duration_h=(
            args.minimum_resolved_duration_h
        ),
        minimum_boundary_change_m=(
            args.minimum_boundary_change_m
        ),
    )

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    result.to_csv(
        args.output,
        index=False,
    )

    print(
        "Quality catalog written:",
        args.output,
    )

    print(
        "Total quality records:",
        len(result),
    )

    if not result.empty:

        print()
        print("Record types:")
        print(
            result.record_type.value_counts()
            .to_string()
        )

        print()
        print("Temporal resolution:")
        print(
            result.temporal_resolution.value_counts()
            .to_string()
        )


if __name__ == "__main__":
    main()
