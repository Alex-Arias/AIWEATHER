#!/usr/bin/env python3
"""Detect experimental Mexican Pacific coastal spectral-wave peaks.

Peak records are diagnostic candidates, not confirmed swell
arrivals, independent wave systems, or operational warnings.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import find_peaks, peak_prominences


SPECTRAL_BANDS = (
    "h1012",
    "h1214",
    "h1417",
    "h1721",
    "h2125",
    "h2530",
)


def validate_input(df):
    """Validate coastal spectral guidance for peak detection."""

    required = {
        "sector",
        "lead_time_hours",
        "swh_mean_m",
        "swh_mean_h2_m2",
        "spectral_sampling_supported",
        "n_spectral_valid_cells",
    }

    for band in SPECTRAL_BANDS:
        required.add(f"{band}_mean_m")
        required.add(f"{band}_mean_h2_m2")

    missing = required.difference(df.columns)

    if missing:
        raise ValueError(
            f"Missing columns: {sorted(missing)}"
        )

    if df.empty:
        raise ValueError("Input dataset is empty")

    for sector, group in df.groupby("sector"):

        leads = np.sort(
            group["lead_time_hours"].to_numpy(
                dtype=float
            )
        )

        if not np.isfinite(leads).all():
            raise ValueError(
                f"{sector}: non-finite lead times"
            )

        if len(leads) < 3:
            raise ValueError(
                f"{sector}: at least three leads required"
            )

        differences = np.diff(leads)

        if np.any(differences <= 0):
            raise ValueError(
                f"{sector}: duplicate lead times"
            )

        if not np.allclose(
            differences,
            differences[0],
        ):
            raise ValueError(
                f"{sector}: irregular forecast cadence"
            )

        values = group[
            ["swh_mean_m", "swh_mean_h2_m2"]
            + [
                f"{band}_mean_m"
                for band in SPECTRAL_BANDS
            ]
            + [
                f"{band}_mean_h2_m2"
                for band in SPECTRAL_BANDS
            ]
        ].to_numpy(dtype=float)

        if not np.isfinite(values).all():
            raise ValueError(
                f"{sector}: non-finite spectral values"
            )

        if np.any(values < 0):
            raise ValueError(
                f"{sector}: negative wave-height diagnostics"
            )

        if np.any(
            group["swh_mean_h2_m2"].to_numpy() <= 0
        ):
            raise ValueError(
                f"{sector}: non-positive total Hs squared"
            )

        supported = group[
            "spectral_sampling_supported"
        ].to_numpy()

        if not np.all(supported == True):
            raise ValueError(
                f"{sector}: unsupported spectral sampling"
            )

        if np.any(
            group["n_spectral_valid_cells"].to_numpy()
            < 3
        ):
            raise ValueError(
                f"{sector}: insufficient spectral cells"
            )

        band_h2 = group[
            [
                f"{band}_mean_h2_m2"
                for band in SPECTRAL_BANDS
            ]
        ].to_numpy(dtype=float)

        total_h2 = group[
            "swh_mean_h2_m2"
        ].to_numpy(dtype=float)

        fraction = (
            band_h2.sum(axis=1) / total_h2
        )

        if np.any(fraction > 1.0 + 1e-8):
            raise ValueError(
                f"{sector}: spectral energy exceeds total"
            )


def detect_spectral_peaks(
    df,
    min_height_m=0.30,
    min_prominence_m=0.10,
    min_separation_h=12.0,
):
    """Return a catalog of candidate spectral-band peaks."""

    validate_input(df)

    parameters = (
        min_height_m,
        min_prominence_m,
        min_separation_h,
    )

    if not all(
        np.isfinite(value) and value > 0
        for value in parameters
    ):
        raise ValueError(
            "Detection parameters must be finite and positive"
        )

    records = []

    for sector, group in df.groupby("sector"):

        group = group.sort_values(
            "lead_time_hours"
        ).reset_index(drop=True)

        leads = group[
            "lead_time_hours"
        ].to_numpy(dtype=float)

        cadence_h = float(
            np.diff(leads)[0]
        )

        minimum_distance = max(
            1,
            int(np.ceil(
                min_separation_h / cadence_h
            )),
        )

        total_h2 = group[
            "swh_mean_h2_m2"
        ].to_numpy(dtype=float)

        for band in SPECTRAL_BANDS:

            values = group[
                f"{band}_mean_m"
            ].to_numpy(dtype=float)

            band_h2 = group[
                f"{band}_mean_h2_m2"
            ].to_numpy(dtype=float)

            peaks, _ = find_peaks(
                values,
                height=min_height_m,
                prominence=min_prominence_m,
                distance=minimum_distance,
            )

            prominences = peak_prominences(
                values,
                peaks,
            )[0]

            for index, prominence in zip(
                peaks,
                prominences,
            ):

                fraction = (
                    band_h2[index] / total_h2[index]
                )

                records.append({
                    "sector": sector,
                    "band": band,
                    "peak_lead_h": float(
                        leads[index]
                    ),
                    "peak_band_hs_m": float(
                        values[index]
                    ),
                    "peak_prominence_m": float(
                        prominence
                    ),
                    "peak_energy_fraction": float(
                        fraction
                    ),
                    "total_hs_at_peak_m": float(
                        group.loc[
                            index,
                            "swh_mean_m",
                        ]
                    ),
                    "left_boundary_peak": False,
                    "right_boundary_peak": False,
                })

            # Boundary maxima are reported separately.
            for index, label in (
                (0, "left"),
                (len(values) - 1, "right"),
            ):

                neighbor = (
                    1 if index == 0
                    else len(values) - 2
                )

                if (
                    values[index] >= min_height_m
                    and values[index] > values[neighbor]
                ):

                    fraction = (
                        band_h2[index] / total_h2[index]
                    )

                    # Compare the boundary with the value
                    # approximately 24 hours inside the forecast.
                    # Require at least four cadence intervals.
                    offset = max(
                        1,
                        int(round(24.0 / cadence_h)),
                    )

                    offset = min(
                        offset,
                        len(values) - 1,
                    )

                    comparison_index = (
                        offset
                        if label == "left"
                        else len(values) - 1 - offset
                    )

                    boundary_change = float(
                        values[index]
                        - values[comparison_index]
                    )

                    comparison_value = float(
                        values[comparison_index]
                    )

                    relative_change = (
                        boundary_change / comparison_value
                        if comparison_value > 0
                        else np.nan
                    )

                    segment = (
                        values[:offset + 1][::-1]
                        if label == "left"
                        else values[-offset - 1:]
                    )

                    consistently_increasing = bool(
                        np.all(np.diff(segment) > 0)
                    )

                    records.append({
                        "sector": sector,
                        "band": band,
                        "peak_lead_h": float(
                            leads[index]
                        ),
                        "peak_band_hs_m": float(
                            values[index]
                        ),
                        "peak_prominence_m": np.nan,
                        "boundary_change_24h_m":
                            boundary_change,
                        "boundary_relative_change_24h":
                            relative_change,
                        "boundary_increase_ge_0p10m":
                            boundary_change >= 0.10,
                        "boundary_monotonic_24h":
                            consistently_increasing,
                        "peak_energy_fraction": float(
                            fraction
                        ),
                        "total_hs_at_peak_m": float(
                            group.loc[
                                index,
                                "swh_mean_m",
                            ]
                        ),
                        "left_boundary_peak":
                            label == "left",
                        "right_boundary_peak":
                            label == "right",
                    })

    columns = [
        "sector",
        "band",
        "peak_lead_h",
        "peak_band_hs_m",
        "peak_prominence_m",
        "boundary_change_24h_m",
        "boundary_relative_change_24h",
        "boundary_increase_ge_0p10m",
        "boundary_monotonic_24h",
        "peak_energy_fraction",
        "total_hs_at_peak_m",
        "left_boundary_peak",
        "right_boundary_peak",
    ]

    return pd.DataFrame(
        records,
        columns=columns,
    ).sort_values(
        ["sector", "band", "peak_lead_h"]
    ).reset_index(drop=True)


def main():

    parser = argparse.ArgumentParser(
        description=(
            "Detect experimental Mexican Pacific "
            "coastal spectral-wave peaks."
        )
    )

    parser.add_argument(
        "--input",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--output",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--min-height-m",
        type=float,
        default=0.30,
    )

    parser.add_argument(
        "--min-prominence-m",
        type=float,
        default=0.10,
    )

    parser.add_argument(
        "--min-separation-h",
        type=float,
        default=12.0,
    )

    args = parser.parse_args()

    df = pd.read_csv(args.input)

    result = detect_spectral_peaks(
        df,
        min_height_m=args.min_height_m,
        min_prominence_m=args.min_prominence_m,
        min_separation_h=args.min_separation_h,
    )

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    result.to_csv(
        args.output,
        index=False,
    )

    print("Spectral peak catalog written:", args.output)
    print("Detected candidate peaks:", len(result))
    print("Sectors:", result.sector.nunique())


if __name__ == "__main__":
    main()
