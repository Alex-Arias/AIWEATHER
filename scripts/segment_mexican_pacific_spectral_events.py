#!/usr/bin/env python3
"""Segment experimental coastal spectral-energy enhancements.

Events are algorithmic spectral-height enhancements, not confirmed
swell arrivals, independent wave systems, or operational warnings.

Only accepted interior peaks seed events. Boundary-only maxima
remain in the original peak catalog for separate evaluation.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from detect_mexican_pacific_spectral_events import (
    SPECTRAL_BANDS,
    validate_input,
)


OUTPUT_COLUMNS = [
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
]


def _boolean(series):
    """Convert catalog boolean values without treating NaN as True."""

    return (
        series.astype("string")
        .str.lower()
        .eq("true")
        .fillna(False)
        .to_numpy(dtype=bool)
    )


def _positive_parameter(value, name):
    """Require a finite positive numeric parameter."""

    if not np.isfinite(value) or value <= 0:
        raise ValueError(
            f"{name} must be finite and positive"
        )


def segment_spectral_events(
    guidance,
    peaks,
    background_window_h=72.0,
    enhancement_fraction=0.25,
    minimum_enhancement_m=0.10,
    merge_valley_ratio=0.80,
):
    """Return candidate spectral-energy enhancement episodes."""

    validate_input(guidance)

    _positive_parameter(
        background_window_h,
        "background_window_h",
    )

    _positive_parameter(
        enhancement_fraction,
        "enhancement_fraction",
    )

    _positive_parameter(
        minimum_enhancement_m,
        "minimum_enhancement_m",
    )

    if (
        not np.isfinite(merge_valley_ratio)
        or not 0 < merge_valley_ratio <= 1
    ):
        raise ValueError(
            "merge_valley_ratio must be in (0, 1]"
        )

    required = {
        "sector",
        "band",
        "peak_lead_h",
        "peak_band_hs_m",
        "left_boundary_peak",
        "right_boundary_peak",
    }

    missing = required.difference(peaks.columns)

    if missing:
        raise ValueError(
            f"Missing peak columns: {sorted(missing)}"
        )

    if peaks.empty:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)

    left = _boolean(peaks["left_boundary_peak"])
    right = _boolean(peaks["right_boundary_peak"])

    interior = peaks.loc[
        ~(left | right)
    ].copy()

    if interior.empty:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)

    records = []

    for (sector, band), candidates in interior.groupby(
        ["sector", "band"],
        sort=True,
    ):

        if band not in SPECTRAL_BANDS:
            raise ValueError(
                f"Unsupported spectral band: {band}"
            )

        series = (
            guidance.loc[
                guidance["sector"] == sector
            ]
            .sort_values("lead_time_hours")
            .reset_index(drop=True)
        )

        if series.empty:
            raise ValueError(
                f"Peak sector absent from guidance: {sector}"
            )

        leads = series[
            "lead_time_hours"
        ].to_numpy(dtype=float)

        heights = series[
            f"{band}_mean_m"
        ].to_numpy(dtype=float)

        candidates = candidates.sort_values(
            "peak_lead_h"
        )

        windows = []

        for peak in candidates.itertuples():

            peak_lead = float(
                peak.peak_lead_h
            )

            matches = np.flatnonzero(
                np.isclose(
                    leads,
                    peak_lead,
                    rtol=0,
                    atol=1e-8,
                )
            )

            if len(matches) != 1:
                raise ValueError(
                    f"{sector}/{band}: "
                    f"peak lead {peak_lead} "
                    "not uniquely present in guidance"
                )

            index = int(matches[0])

            if index == 0 or index == len(leads) - 1:
                raise ValueError(
                    "Interior peak cannot lie at a boundary"
                )

            if not np.isclose(
                heights[index],
                float(peak.peak_band_hs_m),
                rtol=0,
                atol=1e-6,
            ):
                raise ValueError(
                    f"{sector}/{band}: "
                    "peak height disagrees with guidance"
                )

            local = (
                (leads >= peak_lead - background_window_h)
                & (leads <= peak_lead + background_window_h)
            )

            background = float(
                np.min(heights[local])
            )

            enhancement = float(
                heights[index] - background
            )

            threshold = (
                background
                + max(
                    minimum_enhancement_m,
                    enhancement_fraction * enhancement,
                )
            )

            if heights[index] < threshold:
                raise ValueError(
                    f"{sector}/{band}: "
                    "peak does not exceed enhancement threshold"
                )

            active = heights >= threshold

            start = index
            end = index

            while start > 0 and active[start - 1]:
                start -= 1

            while (
                end < len(heights) - 1
                and active[end + 1]
            ):
                end += 1

            windows.append({
                "start": start,
                "end": end,
                "peak_index": index,
                "peak_lead": peak_lead,
                "peak_height": float(heights[index]),
            })

        windows.sort(
            key=lambda item: item["peak_index"]
        )

        clusters = []
        current = [windows[0]]
        cluster_ambiguous = False
        valley_ratios = []

        def finish_cluster():
            clusters.append((
                list(current),
                cluster_ambiguous,
                list(valley_ratios),
            ))

        for window in windows[1:]:

            previous = current[-1]

            between = heights[
                previous["peak_index"]:
                window["peak_index"] + 1
            ]

            valley = float(
                np.min(between)
            )

            smaller_peak = min(
                previous["peak_height"],
                window["peak_height"],
            )

            ratio = (
                valley / smaller_peak
                if smaller_peak > 0
                else np.nan
            )

            overlaps = (
                window["start"]
                <= max(item["end"] for item in current)
            )

            if overlaps and ratio >= merge_valley_ratio:

                current.append(window)
                valley_ratios.append(ratio)

            else:

                if overlaps:
                    cluster_ambiguous = True

                finish_cluster()

                current = [window]
                cluster_ambiguous = overlaps
                valley_ratios = []

        finish_cluster()

        for cluster, ambiguous, ratios in clusters:

            start = min(
                item["start"]
                for item in cluster
            )

            end = max(
                item["end"]
                for item in cluster
            )

            primary = max(
                cluster,
                key=lambda item: item["peak_height"],
            )

            peak_leads = [
                item["peak_lead"]
                for item in cluster
            ]

            records.append({
                "sector": sector,
                "band": band,
                "onset_lead_h": float(leads[start]),
                "end_lead_h": float(leads[end]),
                "duration_h": float(
                    leads[end] - leads[start]
                ),
                "primary_peak_lead_h": (
                    primary["peak_lead"]
                ),
                "primary_peak_band_hs_m": (
                    primary["peak_height"]
                ),
                "n_constituent_peaks": len(cluster),
                "constituent_peak_leads_h": ";".join(
                    f"{value:g}"
                    for value in peak_leads
                ),
                "left_censored": start == 0,
                "right_censored": end == len(leads) - 1,
                "ambiguous_overlap": ambiguous,
                "minimum_interpeak_valley_ratio": (
                    min(ratios)
                    if ratios
                    else np.nan
                ),
            })

    return (
        pd.DataFrame(
            records,
            columns=OUTPUT_COLUMNS,
        )
        .sort_values(
            ["sector", "band", "onset_lead_h"]
        )
        .reset_index(drop=True)
    )


def main():
    parser = argparse.ArgumentParser(
        description=__doc__
    )

    parser.add_argument(
        "--input",
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
        "--background-window-h",
        type=float,
        default=72.0,
    )

    parser.add_argument(
        "--enhancement-fraction",
        type=float,
        default=0.25,
    )

    parser.add_argument(
        "--minimum-enhancement-m",
        type=float,
        default=0.10,
    )

    parser.add_argument(
        "--merge-valley-ratio",
        type=float,
        default=0.80,
    )

    args = parser.parse_args()

    guidance = pd.read_csv(args.input)
    peaks = pd.read_csv(args.peaks)

    events = segment_spectral_events(
        guidance,
        peaks,
        background_window_h=args.background_window_h,
        enhancement_fraction=args.enhancement_fraction,
        minimum_enhancement_m=args.minimum_enhancement_m,
        merge_valley_ratio=args.merge_valley_ratio,
    )

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    events.to_csv(
        args.output,
        index=False,
    )

    print(
        "Spectral event catalog written:",
        args.output,
    )

    print("Candidate episodes:", len(events))

    if not events.empty:
        print(
            "Sectors:",
            events["sector"].nunique(),
        )

        print(
            "Episodes with multiple peaks:",
            int(
                (
                    events["n_constituent_peaks"] > 1
                ).sum()
            ),
        )

        print(
            "Ambiguous overlaps:",
            int(
                events["ambiguous_overlap"].sum()
            ),
        )


if __name__ == "__main__":
    main()
