#!/usr/bin/env python3
"""
Build Mexican Pacific coastal spectral-wave guidance from AIFS2.

This is an experimental diagnostic product, not a warning system.

It uses the coastal anchors, distance calculation, and sampling
conventions from build_mexican_pacific_wave_guidance.py.

Period-band significant wave heights:
    h1012 : 10-12 s
    h1214 : 12-14 s
    h1417 : 14-17 s
    h1721 : 17-21 s
    h2125 : 21-25 s
    h2530 : 25-30 s

The period bands do not independently identify wave direction,
generation source, or tropical-cyclone attribution.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from build_mexican_pacific_wave_guidance import (
    COASTAL_ANCHORS,
    DEFAULT_RADIUS_KM,
    haversine_km,
    normalize_longitude,
    lead_time_hours,
    valid_time_from,
    validate_dataset,
)


SPECTRAL_BANDS = (
    "h1012",
    "h1214",
    "h1417",
    "h1721",
    "h2125",
    "h2530",
)

BASE_FIELDS = (
    "swh",
    "mwp",
    "cos_mwd",
    "sin_mwd",
)


def validate_spectral_dataset(ds):
    """Validate the original wave fields and spectral bands."""

    validate_dataset(ds)

    missing = [
        name
        for name in SPECTRAL_BANDS
        if name not in ds
    ]

    if missing:
        raise ValueError(
            "Missing spectral variables: "
            + ", ".join(missing)
        )

    if ds.sizes["time"] != 1:
        raise ValueError(
            "Expected exactly one initialization time"
        )


def build_spectral_guidance(
    ds,
    radius_km=DEFAULT_RADIUS_KM,
    anchors=COASTAL_ANCHORS,
):
    """Build period-resolved coastal wave diagnostics."""

    validate_spectral_dataset(ds)

    if not np.isfinite(radius_km) or radius_km <= 0:
        raise ValueError(
            "radius_km must be finite and positive"
        )

    lat = np.asarray(
        ds["lat"].values,
        dtype=float,
    )

    lon = normalize_longitude(
        ds["lon"].values
    )

    initialization = ds["time"].values[0]

    spatial_indices = {}

    for sector, anchor, alat, alon in anchors:

        distance = haversine_km(
            alat,
            alon,
            lat[:, None],
            lon[None, :],
        )

        iy, ix = np.where(
            distance <= radius_km
        )

        spatial_indices[sector] = (
            iy,
            ix,
        )

    rows = []

    for lead_index, lead in enumerate(
        ds["lead_time"].values
    ):

        lead_hours = lead_time_hours(lead)

        valid_time = valid_time_from(
            initialization,
            lead,
        )

        # Load each field once per forecast lead.
        fields = {}

        for name in BASE_FIELDS + SPECTRAL_BANDS:

            fields[name] = np.asarray(
                ds[name].isel(
                    time=0,
                    lead_time=lead_index,
                ).values,
                dtype=float,
            )

        for sector, anchor, alat, alon in anchors:

            iy, ix = spatial_indices[sector]

            sampled = {
                name: values[iy, ix]
                for name, values in fields.items()
            }

            # Exactly reproduce the original wave-guidance
            # valid-cell definition.
            original_valid = np.ones(
                len(iy),
                dtype=bool,
            )

            for name in BASE_FIELDS:
                original_valid &= np.isfinite(
                    sampled[name]
                )

            # Spectral completeness is evaluated separately.
            spectral_valid = original_valid.copy()

            for name in SPECTRAL_BANDS:
                spectral_valid &= np.isfinite(
                    sampled[name]
                )

            n_original = int(
                original_valid.sum()
            )

            n_spectral = int(
                spectral_valid.sum()
            )

            row = {
                "sector": sector,
                "anchor_name": anchor,
                "anchor_latitude": float(alat),
                "anchor_longitude": float(alon),
                "sampling_radius_km": float(radius_km),
                "valid_time": pd.Timestamp(valid_time),
                "lead_time_hours": lead_hours,
                "n_candidate_cells": len(iy),
                "n_original_valid_cells": n_original,
                "n_spectral_valid_cells": n_spectral,
                "n_missing_spectral_cells":
                    n_original - n_spectral,
                "minimum_cells": 3,
                "sampling_supported":
                    bool(n_original >= 3),
                "spectral_sampling_supported":
                    bool(n_spectral >= 3),
            }

            # Initialize all numerical diagnostics.
            for name in SPECTRAL_BANDS:
                row[f"{name}_mean_m"] = np.nan
                row[f"{name}_mean_h2_m2"] = np.nan

            row.update({
                "swh_mean_m": np.nan,
                "swh_max_m": np.nan,
                "swh_mean_h2_m2": np.nan,
                "combined_band_hs_m": np.nan,
                "band_energy_fraction": np.nan,
                "fraction_cells_energy_excess": np.nan,
                "max_cell_energy_ratio": np.nan,
            })

            if n_spectral >= 3:

                hs = sampled["swh"][
                    spectral_valid
                ]

                total_h2 = float(
                    np.mean(hs ** 2)
                )

                row["swh_mean_m"] = float(
                    np.mean(hs)
                )

                row["swh_max_m"] = float(
                    np.max(hs)
                )

                row["swh_mean_h2_m2"] = total_h2

                combined_cell_h2 = np.zeros_like(
                    hs,
                    dtype=float,
                )

                for name in SPECTRAL_BANDS:

                    values = sampled[name][
                        spectral_valid
                    ]

                    row[f"{name}_mean_m"] = float(
                        np.mean(values)
                    )

                    row[f"{name}_mean_h2_m2"] = float(
                        np.mean(values ** 2)
                    )

                    combined_cell_h2 += values ** 2

                mean_band_h2 = float(
                    np.mean(combined_cell_h2)
                )

                row["combined_band_hs_m"] = float(
                    np.sqrt(mean_band_h2)
                )

                if total_h2 > 0:

                    row["band_energy_fraction"] = (
                        mean_band_h2 / total_h2
                    )

                row["fraction_cells_energy_excess"] = (
                    float(
                        np.mean(
                            combined_cell_h2 > hs ** 2
                        )
                    )
                )

                # Avoid division by zero at calm cells.
                positive_hs = hs > 0

                if np.any(positive_hs):

                    ratios = (
                        combined_cell_h2[positive_hs]
                        / hs[positive_hs] ** 2
                    )

                    row["max_cell_energy_ratio"] = (
                        float(np.max(ratios))
                    )

            rows.append(row)

    return (
        pd.DataFrame(rows)
        .sort_values(
            ["lead_time_hours", "sector"]
        )
        .reset_index(drop=True)
    )


def parse_args():

    parser = argparse.ArgumentParser(
        description=(
            "Build experimental Mexican Pacific "
            "coastal spectral-wave guidance."
        )
    )

    parser.add_argument(
        "--forecast",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--output",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--radius-km",
        type=float,
        default=DEFAULT_RADIUS_KM,
    )

    return parser.parse_args()


def main():

    args = parse_args()

    ds = xr.open_zarr(
        args.forecast
    )

    result = build_spectral_guidance(
        ds,
        radius_km=args.radius_km,
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
        "Spectral guidance written:",
        args.output,
    )

    print(
        "Rows:",
        len(result),
    )

    print(
        "Sectors:",
        result.sector.nunique(),
    )

    print(
        "Supported spectral rows:",
        int(
            result.spectral_sampling_supported.sum()
        ),
    )

    print(
        "Rows with missing spectral cells:",
        int(
            (
                result.n_missing_spectral_cells > 0
            ).sum()
        ),
    )

    print(
        "Rows with energy fraction > 1:",
        int(
            (
                result.band_energy_fraction > 1
            ).sum()
        ),
    )


if __name__ == "__main__":
    main()
