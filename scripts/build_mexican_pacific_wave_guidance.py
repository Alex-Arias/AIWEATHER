#!/usr/bin/env python3
"""
Build objective Mexican Pacific coastal-sector wave guidance from an AIFS2
forecast.zarr store.

This product is a wave-guidance interface, not a warning system.

It does NOT assign:
    - outlook/advisory/warning categories,
    - hazard thresholds,
    - calibrated probabilities,
    - TC confidence categories.

Wave-direction convention
-------------------------
AIFS2/Earth2Studio stores cos_mwd and sin_mwd derived directly from
ECMWF mean wave direction (MWD).

The reconstructed MWD is therefore retained explicitly as the
wave-from bearing:

    mwd_from_deg

The corresponding propagation bearing is:

    wave_propagation_deg = (mwd_from_deg + 180) % 360

Directional means are calculated from spatially averaged sine and
cosine components rather than by arithmetic averaging of angles.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr


DEFAULT_RADIUS_KM = 100.0


COASTAL_ANCHORS = (
    ("BC North", "Ensenada", 31.86, -116.62),
    ("BC Central", "Punta Eugenia", 27.85, -115.08),
    ("Los Cabos", "Cabo San Lucas", 22.88, -109.91),
    ("Sinaloa", "Mazatlan", 23.25, -106.42),
    ("Nayarit", "San Blas", 21.54, -105.29),
    ("Jalisco/Nayarit", "Puerto Vallarta", 20.65, -105.25),
    ("Colima", "Manzanillo", 19.05, -104.32),
    ("Michoacan", "Lazaro Cardenas", 17.96, -102.20),
    ("Guerrero", "Acapulco", 16.85, -99.90),
    ("Oaxaca West", "Puerto Escondido", 15.86, -97.07),
    ("Oaxaca East", "Salina Cruz", 16.17, -95.20),
)


REQUIRED_VARIABLES = (
    "swh",
    "mwp",
    "cos_mwd",
    "sin_mwd",
)


def normalize_longitude(lon):
    """Normalize longitude to [-180, 180)."""

    lon = np.asarray(
        lon,
        dtype=float,
    )

    return (
        lon + 180.0
    ) % 360.0 - 180.0


def haversine_km(
    lat1,
    lon1,
    lat2,
    lon2,
):
    """Great-circle distance in km."""

    radius_km = 6371.0

    lat1 = np.radians(lat1)
    lat2 = np.radians(lat2)

    dlat = lat2 - lat1

    dlon = np.radians(
        normalize_longitude(
            np.asarray(lon2)
            - float(lon1)
        )
    )

    a = (
        np.sin(dlat / 2.0) ** 2
        + np.cos(lat1)
        * np.cos(lat2)
        * np.sin(dlon / 2.0) ** 2
    )

    return (
        2.0
        * radius_km
        * np.arcsin(
            np.sqrt(a)
        )
    )


def circular_mwd_from_components(
    sin_values,
    cos_values,
):
    """
    Return circular-mean ECMWF MWD from sine/cosine components.

    NaN is returned if no paired finite directional components exist.
    """

    sin_values = np.asarray(
        sin_values,
        dtype=float,
    )

    cos_values = np.asarray(
        cos_values,
        dtype=float,
    )

    finite = (
        np.isfinite(sin_values)
        & np.isfinite(cos_values)
    )

    if not finite.any():
        return np.nan

    mean_sin = np.mean(
        sin_values[finite]
    )

    mean_cos = np.mean(
        cos_values[finite]
    )

    if (
        np.isclose(mean_sin, 0.0)
        and np.isclose(mean_cos, 0.0)
    ):
        return np.nan

    return float(
        (
            np.degrees(
                np.arctan2(
                    mean_sin,
                    mean_cos,
                )
            )
            + 360.0
        )
        % 360.0
    )


def propagation_from_mwd(
    mwd_from_deg,
):
    """Convert ECMWF wave-from bearing to propagation bearing."""

    if not np.isfinite(
        mwd_from_deg
    ):
        return np.nan

    return float(
        (
            mwd_from_deg
            + 180.0
        )
        % 360.0
    )


def validate_dataset(ds):
    """Validate required AIFS2 wave fields and coordinates."""

    missing = [
        name
        for name in REQUIRED_VARIABLES
        if name not in ds
    ]

    if missing:
        raise ValueError(
            "Missing required wave variables: "
            + ", ".join(missing)
        )

    for coord in (
        "time",
        "lead_time",
        "lat",
        "lon",
    ):
        if coord not in ds.coords:
            raise ValueError(
                f"Missing required coordinate: {coord}"
            )


def lead_time_hours(value):
    """Convert a timedelta-like lead to integer hours."""

    value = np.asarray(value)

    return int(
        value.astype(
            "timedelta64[h]"
        ).astype(int)
    )


def valid_time_from(
    initialization,
    lead,
):
    """Return forecast valid time."""

    initialization = np.datetime64(
        initialization
    )

    lead = np.asarray(
        lead
    ).astype(
        "timedelta64[ns]"
    )

    return initialization + lead


def build_guidance(
    ds,
    radius_km=DEFAULT_RADIUS_KM,
    anchors=COASTAL_ANCHORS,
):
    """Build the coastal wave-guidance dataframe."""

    validate_dataset(ds)

    if radius_km <= 0.0:
        raise ValueError(
            "radius_km must be positive"
        )

    if ds.sizes["time"] != 1:
        raise ValueError(
            "Expected exactly one initialization time"
        )

    lat = np.asarray(
        ds["lat"].values,
        dtype=float,
    )

    lon = normalize_longitude(
        ds["lon"].values
    )

    lat_grid, lon_grid = np.meshgrid(
        lat,
        lon,
        indexing="ij",
    )

    initialization = np.asarray(
        ds["time"].values
    ).reshape(-1)[0]

    spatial_masks = {}

    for (
        sector,
        anchor_name,
        anchor_lat,
        anchor_lon,
    ) in anchors:

        distance = haversine_km(
            anchor_lat,
            anchor_lon,
            lat_grid,
            lon_grid,
        )

        spatial_masks[
            (
                sector,
                anchor_name,
                anchor_lat,
                anchor_lon,
            )
        ] = (
            distance <= radius_km,
            distance,
        )

    rows = []

    for lead_index, lead in enumerate(
        ds["lead_time"].values
    ):

        lead_hours = lead_time_hours(
            lead
        )

        valid_time = valid_time_from(
            initialization,
            lead,
        )

        swh = np.asarray(
            ds["swh"]
            .isel(
                time=0,
                lead_time=lead_index,
            )
            .values,
            dtype=float,
        )

        mwp = np.asarray(
            ds["mwp"]
            .isel(
                time=0,
                lead_time=lead_index,
            )
            .values,
            dtype=float,
        )

        cos_mwd = np.asarray(
            ds["cos_mwd"]
            .isel(
                time=0,
                lead_time=lead_index,
            )
            .values,
            dtype=float,
        )

        sin_mwd = np.asarray(
            ds["sin_mwd"]
            .isel(
                time=0,
                lead_time=lead_index,
            )
            .values,
            dtype=float,
        )

        for (
            key,
            (
                spatial_mask,
                distance,
            ),
        ) in spatial_masks.items():

            (
                sector,
                anchor_name,
                anchor_lat,
                anchor_lon,
            ) = key

            valid = (
                spatial_mask
                & np.isfinite(swh)
                & np.isfinite(mwp)
                & np.isfinite(cos_mwd)
                & np.isfinite(sin_mwd)
            )

            n_valid = int(
                valid.sum()
            )

            if n_valid:
                swh_values = swh[
                    valid
                ]

                mwp_values = mwp[
                    valid
                ]

                mwd_from = (
                    circular_mwd_from_components(
                        sin_mwd[valid],
                        cos_mwd[valid],
                    )
                )

                propagation = (
                    propagation_from_mwd(
                        mwd_from
                    )
                )

                nearest_flat = np.argmin(
                    np.where(
                        valid,
                        distance,
                        np.inf,
                    )
                )

                iy, ix = np.unravel_index(
                    nearest_flat,
                    valid.shape,
                )

                sample_lat = float(
                    lat_grid[iy, ix]
                )

                sample_lon = float(
                    lon_grid[iy, ix]
                )

                nearest_distance = float(
                    distance[iy, ix]
                )

                swh_mean = float(
                    np.mean(
                        swh_values
                    )
                )

                swh_max = float(
                    np.max(
                        swh_values
                    )
                )

                mwp_mean = float(
                    np.mean(
                        mwp_values
                    )
                )

                available = True

            else:
                sample_lat = np.nan
                sample_lon = np.nan
                nearest_distance = np.nan
                swh_mean = np.nan
                swh_max = np.nan
                mwp_mean = np.nan
                mwd_from = np.nan
                propagation = np.nan
                available = False

            rows.append(
                {
                    "sector":
                        sector,

                    "anchor_name":
                        anchor_name,

                    "anchor_latitude":
                        float(anchor_lat),

                    "anchor_longitude":
                        float(anchor_lon),

                    "sampling_radius_km":
                        float(radius_km),

                    "valid_time":
                        pd.Timestamp(
                            valid_time
                        ),

                    "lead_time_hours":
                        lead_hours,

                    "sample_latitude":
                        sample_lat,

                    "sample_longitude":
                        sample_lon,

                    "nearest_valid_distance_km":
                        nearest_distance,

                    "n_valid_cells":
                        n_valid,

                    "minimum_cells":
                        3,

                    "sampling_supported":
                        bool(n_valid >= 3),

                    "swh_mean_m":
                        swh_mean,

                    "swh_max_m":
                        swh_max,

                    "mwp_mean_s":
                        mwp_mean,

                    "mwd_from_deg":
                        mwd_from,

                    "wave_propagation_deg":
                        propagation,

                    "wave_data_available":
                        available,
                }
            )

    result = pd.DataFrame(
        rows
    )

    return result.sort_values(
        [
            "lead_time_hours",
            "sector",
        ]
    ).reset_index(
        drop=True
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Build objective Mexican Pacific "
            "coastal-sector wave guidance from an "
            "AIFS2 forecast.zarr store."
        )
    )

    parser.add_argument(
        "--forecast",
        required=True,
        help=(
            "Path to the AIFS2 "
            "forecast.zarr store."
        ),
    )

    parser.add_argument(
        "--output",
        required=True,
        help="Output CSV path.",
    )

    parser.add_argument(
        "--radius-km",
        type=float,
        default=DEFAULT_RADIUS_KM,
        help=(
            "Sampling radius around each "
            "coastal anchor in km "
            f"(default: {DEFAULT_RADIUS_KM:g})."
        ),
    )

    return parser.parse_args()


def main():
    args = parse_args()

    forecast = Path(
        args.forecast
    )

    if not forecast.exists():
        raise FileNotFoundError(
            forecast
        )

    output = Path(
        args.output
    )

    ds = xr.open_zarr(
        forecast
    )

    guidance = build_guidance(
        ds,
        radius_km=args.radius_km,
    )

    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    guidance.to_csv(
        output,
        index=False,
    )

    print(
        f"Wrote {len(guidance)} rows to {output}"
    )

    print(
        "Lead-time range:",
        guidance["lead_time_hours"].min(),
        "to",
        guidance["lead_time_hours"].max(),
        "h",
    )

    print(
        "Sectors:",
        guidance["sector"].nunique(),
    )

    print(
        "Unavailable rows:",
        int(
            (
                ~guidance[
                    "wave_data_available"
                ]
            ).sum()
        ),
    )

    print(
        "Rows with fewer than 3 valid cells:",
        int(
            (
                ~guidance[
                    "sampling_supported"
                ]
            ).sum()
        ),
    )


if __name__ == "__main__":
    main()
