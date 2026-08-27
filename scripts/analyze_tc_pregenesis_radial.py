#!/usr/bin/env python3
"""
Generic radial tropical-cyclone organization diagnostic.

Compares one or more AIWeather model forecasts against an IBTrACS
reference track and diagnoses the development of a coherent tropical
vortex near the observed storm center.

The organized-vortex thresholds are exploratory structural criteria,
not an official tropical-cyclone genesis definition.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from aiweather.verification.ibtracs import read_ibtracs_csv
from aiweather.verification.best_track import best_track_to_records


DEFAULT_MODELS = [
    "graphcast",
    "aifs2",
    "pangu3",
]

MODEL_LABELS = {
    "graphcast": "GraphCast",
    "aifs2": "AIFS2",
    "pangu3": "Pangu3",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Analyze radial tropical-cyclone organization "
            "against an IBTrACS reference track."
        )
    )

    parser.add_argument(
        "--sid",
        required=True,
        help="IBTrACS storm SID.",
    )

    parser.add_argument(
        "--init-time",
        required=True,
        help=(
            "Forecast initialization time as "
            "YYYYMMDDTHHMMSS or YYYY-MM-DDTHH:MM:SS."
        ),
    )

    parser.add_argument(
        "--storm-name",
        default=None,
        help="Optional storm name for labels/output.",
    )

    parser.add_argument(
        "--models",
        nargs="+",
        default=DEFAULT_MODELS,
        help=(
            "Models to analyze. "
            "Default: graphcast aifs2 pangu3."
        ),
    )

    parser.add_argument(
        "--lead-min",
        type=int,
        default=0,
        help="Minimum diagnostic lead time in hours. Default: 0.",
    )

    parser.add_argument(
        "--lead-max",
        type=int,
        default=120,
        help="Maximum diagnostic lead time in hours. Default: 120.",
    )

    parser.add_argument(
        "--lead-step",
        type=int,
        default=6,
        help="Diagnostic lead-time spacing in hours. Default: 6.",
    )

    parser.add_argument(
        "--core-radius-km",
        type=float,
        default=200.0,
        help="Core diagnostic radius in km. Default: 200.",
    )

    parser.add_argument(
        "--environment-inner-radius-km",
        type=float,
        default=300.0,
        help=(
            "Inner radius of environmental pressure annulus. "
            "Default: 300 km."
        ),
    )

    parser.add_argument(
        "--environment-outer-radius-km",
        type=float,
        default=500.0,
        help=(
            "Outer radius of environmental pressure annulus. "
            "Default: 500 km."
        ),
    )

    parser.add_argument(
        "--maximum-center-error-km",
        type=float,
        default=200.0,
        help=(
            "Maximum pressure/vorticity center error. "
            "Default: 200 km."
        ),
    )

    parser.add_argument(
        "--maximum-center-separation-km",
        type=float,
        default=200.0,
        help=(
            "Maximum pressure-vorticity center separation. "
            "Default: 200 km."
        ),
    )

    parser.add_argument(
        "--minimum-zeta850",
        type=float,
        default=15.0e-5,
        help=(
            "Minimum maximum 850-hPa relative vorticity "
            "in s^-1. Default: 15e-5."
        ),
    )

    parser.add_argument(
        "--minimum-pressure-deficit-hpa",
        type=float,
        default=2.0,
        help=(
            "Minimum environmental-minus-core pressure deficit. "
            "Default: 2 hPa."
        ),
    )

    parser.add_argument(
        "--ibtracs",
        default=(
            "data/verification/ibtracs/"
            "ibtracs.EP.list.v04r01.csv"
        ),
        help="Path to IBTrACS CSV.",
    )

    parser.add_argument(
        "--output",
        default=None,
        help=(
            "Output CSV path. If omitted, a path under "
            "results/verification/comparison is generated."
        ),
    )

    return parser.parse_args()


def normalize_init_time(value: str):
    value = value.strip()

    if "T" in value and "-" not in value:
        return (
            f"{value[0:4]}-{value[4:6]}-{value[6:8]}"
            f"T{value[9:11]}:{value[11:13]}:{value[13:15]}"
        )

    return value


def init_directory_name(value: str):
    normalized = np.datetime64(
        normalize_init_time(value)
    ).astype("datetime64[s]")

    text = str(normalized)

    return (
        text
        .replace("-", "")
        .replace(":", "")
    )


def forecast_lead_hours(dataset):
    values = dataset["lead_time"].values

    if np.issubdtype(
        values.dtype,
        np.timedelta64,
    ):
        return (
            values
            .astype("timedelta64[h]")
            .astype(int)
        )

    return values.astype(int)


def signed_longitude(longitude):
    return (
        (longitude + 180.0)
        % 360.0
    ) - 180.0


def haversine_distance_km(
    latitude,
    longitude,
    center_latitude,
    center_longitude,
):
    radius = 6371.0

    lat1 = np.deg2rad(center_latitude)
    lon1 = np.deg2rad(center_longitude)

    lat2 = np.deg2rad(latitude)
    lon2 = np.deg2rad(longitude)

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = (
        np.sin(dlat / 2.0) ** 2
        + np.cos(lat1)
        * np.cos(lat2)
        * np.sin(dlon / 2.0) ** 2
    )

    return (
        2.0
        * radius
        * np.arcsin(np.sqrt(a))
    )


def horizontal_vorticity(u, v):
    earth_radius = 6_371_000.0

    lat = np.deg2rad(
        u["lat"].values
    )

    lon = np.deg2rad(
        u["lon"].values
    )

    u_values = np.asarray(
        u.values,
        dtype=float,
    )

    v_values = np.asarray(
        v.values,
        dtype=float,
    )

    y = earth_radius * lat

    du_dy = np.gradient(
        u_values,
        y,
        axis=-2,
        edge_order=2,
    )

    dv_dlambda = np.gradient(
        v_values,
        lon,
        axis=-1,
        edge_order=2,
    )

    dx_factor = (
        earth_radius
        * np.cos(lat)[:, None]
    )

    dv_dx = (
        dv_dlambda
        / dx_factor
    )

    return xr.DataArray(
        dv_dx - du_dy,
        coords=u.coords,
        dims=u.dims,
        name="zeta850",
        attrs={
            "units": "s-1",
        },
    )


def local_subset(
    dataset,
    center_latitude,
    center_longitude,
    half_width_degrees=6.0,
):
    lon_values = signed_longitude(
        dataset["lon"].values
    )

    lat_values = dataset[
        "lat"
    ].values

    lat_index = np.where(
        (
            lat_values
            >= center_latitude - half_width_degrees
        )
        & (
            lat_values
            <= center_latitude + half_width_degrees
        )
    )[0]

    lon_index = np.where(
        (
            lon_values
            >= center_longitude - half_width_degrees
        )
        & (
            lon_values
            <= center_longitude + half_width_degrees
        )
    )[0]

    return dataset.isel(
        lat=lat_index,
        lon=lon_index,
    )


def radial_distance_field(
    dataset,
    center_latitude,
    center_longitude,
):
    lat = dataset["lat"].values

    lon = signed_longitude(
        dataset["lon"].values
    )

    lon2d, lat2d = np.meshgrid(
        lon,
        lat,
    )

    distance = haversine_distance_km(
        lat2d,
        lon2d,
        center_latitude,
        center_longitude,
    )

    return xr.DataArray(
        distance,
        coords={
            "lat": dataset["lat"],
            "lon": dataset["lon"],
        },
        dims=(
            "lat",
            "lon",
        ),
    )


def extrema_location(field, mode):
    values = np.asarray(
        field.values,
        dtype=float,
    )

    if np.all(~np.isfinite(values)):
        raise ValueError(
            "No finite values available for extrema search."
        )

    if mode == "min":
        flat_index = np.nanargmin(values)
    elif mode == "max":
        flat_index = np.nanargmax(values)
    else:
        raise ValueError(
            "mode must be 'min' or 'max'."
        )

    index = np.unravel_index(
        flat_index,
        values.shape,
    )

    latitude = float(
        field["lat"].values[
            index[-2]
        ]
    )

    longitude = float(
        signed_longitude(
            field["lon"].values[
                index[-1]
            ]
        )
    )

    value = float(
        values[index]
    )

    return (
        value,
        latitude,
        longitude,
    )


def validate_args(args):
    if args.lead_step <= 0:
        raise ValueError(
            "--lead-step must be positive."
        )

    if args.lead_min < 0:
        raise ValueError(
            "--lead-min cannot be negative."
        )

    if args.lead_max < args.lead_min:
        raise ValueError(
            "--lead-max must be >= --lead-min."
        )

    if args.core_radius_km <= 0:
        raise ValueError(
            "--core-radius-km must be positive."
        )

    if (
        args.environment_inner_radius_km
        <= args.core_radius_km
    ):
        raise ValueError(
            "Environmental inner radius must be "
            "greater than core radius."
        )

    if (
        args.environment_outer_radius_km
        <= args.environment_inner_radius_km
    ):
        raise ValueError(
            "Environmental outer radius must be "
            "greater than inner radius."
        )


def main():
    args = parse_args()

    validate_args(args)

    initialization_time = normalize_init_time(
        args.init_time
    )

    init_dir = init_directory_name(
        args.init_time
    )

    storm_name = (
        args.storm_name
        if args.storm_name
        else args.sid
    )

    diagnostic_leads = list(
        range(
            args.lead_min,
            args.lead_max + 1,
            args.lead_step,
        )
    )

    forecast_paths = {
        model: Path(
            f"outputs/{model}/"
            f"{init_dir}/forecast.zarr"
        )
        for model in args.models
    }

    if args.output is None:
        output_dir = Path(
            "results/verification/"
            "comparison/"
            f"{storm_name.lower()}_pregenesis"
        )

        output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        output_csv = (
            output_dir
            / (
                f"{storm_name.lower()}_"
                f"{init_dir}_radial_diagnostic.csv"
            )
        )

    else:
        output_csv = Path(
            args.output
        )

        output_csv.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

    best_track_points = read_ibtracs_csv(
        Path(args.ibtracs),
        sid=args.sid,
    )

    reference_records = best_track_to_records(
        best_track_points,
        initialization_time=(
            initialization_time
        ),
    )

    reference_by_lead = {
        record.lead_time_hours: record
        for record in reference_records
    }

    reference_leads = sorted(
        reference_by_lead
    )

    if not reference_leads:
        raise ValueError(
            f"No IBTrACS records found for SID {args.sid}."
        )

    reference_lead_min = min(
        reference_leads
    )

    reference_lead_max = max(
        reference_leads
    )

    print("=" * 112)
    print(
        f"{storm_name.upper()} "
        "RADIAL TROPICAL-VORTEX STRUCTURE DIAGNOSTIC"
    )
    print("=" * 112)

    print(
        "SID                       :",
        args.sid,
    )

    print(
        "Initialization            :",
        initialization_time,
    )

    print(
        "Core radius               :",
        args.core_radius_km,
        "km",
    )

    print(
        "Environmental annulus     :",
        args.environment_inner_radius_km,
        "-",
        args.environment_outer_radius_km,
        "km",
    )

    print(
        "IBTrACS reference range   :",
        reference_lead_min,
        "-",
        reference_lead_max,
        "h",
    )

    print(
        "Requested diagnostic range:",
        args.lead_min,
        "-",
        args.lead_max,
        "h",
    )

    if (
        args.lead_max
        > reference_lead_max
    ):
        print(
            "Note                      : "
            "leads beyond the IBTrACS record "
            "are excluded from reference-centered "
            "diagnostics."
        )

    print()
    print("Organized-vortex criteria:")
    print(
        "  center error            <=",
        args.maximum_center_error_km,
        "km",
    )
    print(
        "  pressure-zeta separation <=",
        args.maximum_center_separation_km,
        "km",
    )
    print(
        "  maximum zeta850         >=",
        f"{args.minimum_zeta850 * 1.0e5:.1f}",
        "x10^-5 s^-1",
    )
    print(
        "  pressure deficit        >=",
        args.minimum_pressure_deficit_hpa,
        "hPa",
    )

    rows = []

    for model, forecast_path in forecast_paths.items():

        if not forecast_path.exists():
            print()
            print("=" * 112)
            print(
                MODEL_LABELS.get(
                    model,
                    model,
                ).upper()
            )
            print("=" * 112)

            print(
                "Forecast not found:",
                forecast_path,
            )

            continue

        ds = xr.open_zarr(
            forecast_path,
            consolidated=False,
        )

        required_variables = [
            "msl",
            "u10m",
            "v10m",
            "u850",
            "v850",
        ]

        missing_variables = [
            variable
            for variable in required_variables
            if variable not in ds
        ]

        if missing_variables:
            ds.close()

            raise ValueError(
                f"{model} forecast is missing required "
                f"variables: {missing_variables}"
            )

        leads = forecast_lead_hours(
            ds
        )

        print()
        print("=" * 112)
        print(
            MODEL_LABELS.get(
                model,
                model,
            ).upper()
        )
        print("=" * 112)

        for lead in diagnostic_leads:

            if lead not in leads:
                continue

            if lead not in reference_by_lead:
                continue

            reference = reference_by_lead[
                lead
            ]

            obs_lat = float(
                reference.latitude
            )

            obs_lon = float(
                reference.longitude
            )

            lead_index = int(
                np.where(
                    leads == lead
                )[0][0]
            )

            snapshot = ds.isel(
                time=0,
                lead_time=lead_index,
            )

            regional = local_subset(
                snapshot,
                obs_lat,
                obs_lon,
            )

            radius = radial_distance_field(
                regional,
                obs_lat,
                obs_lon,
            )

            core = (
                radius
                <= args.core_radius_km
            )

            environment = (
                (
                    radius
                    >= args.environment_inner_radius_km
                )
                & (
                    radius
                    <= args.environment_outer_radius_km
                )
            )

            msl = (
                regional["msl"]
                / 100.0
            )

            core_msl = msl.where(
                core
            )

            (
                minimum_mslp,
                pressure_latitude,
                pressure_longitude,
            ) = extrema_location(
                core_msl,
                "min",
            )

            pressure_center_error = float(
                haversine_distance_km(
                    pressure_latitude,
                    pressure_longitude,
                    obs_lat,
                    obs_lon,
                )
            )

            environment_mslp = float(
                msl.where(
                    environment
                )
                .mean(
                    skipna=True
                )
                .values
            )

            pressure_deficit = (
                environment_mslp
                - minimum_mslp
            )

            wind10 = np.hypot(
                regional["u10m"],
                regional["v10m"],
            )

            maximum_wind = float(
                wind10.where(
                    core
                )
                .max(
                    skipna=True
                )
                .values
            )

            zeta850 = horizontal_vorticity(
                regional["u850"],
                regional["v850"],
            )

            core_zeta = zeta850.where(
                core
            )

            (
                maximum_zeta,
                zeta_latitude,
                zeta_longitude,
            ) = extrema_location(
                core_zeta,
                "max",
            )

            zeta_center_error = float(
                haversine_distance_km(
                    zeta_latitude,
                    zeta_longitude,
                    obs_lat,
                    obs_lon,
                )
            )

            positive_zeta = (
                core_zeta.where(
                    core_zeta > 0.0
                )
            )

            mean_positive_zeta = float(
                positive_zeta.mean(
                    skipna=True
                ).values
            )

            pressure_zeta_separation = float(
                haversine_distance_km(
                    pressure_latitude,
                    pressure_longitude,
                    zeta_latitude,
                    zeta_longitude,
                )
            )

            organized_vortex = (
                pressure_center_error
                <= args.maximum_center_error_km
                and zeta_center_error
                <= args.maximum_center_error_km
                and pressure_zeta_separation
                <= args.maximum_center_separation_km
                and maximum_zeta
                >= args.minimum_zeta850
                and pressure_deficit
                >= args.minimum_pressure_deficit_hpa
            )

            valid_time = (
                np.datetime64(
                    initialization_time
                )
                + np.timedelta64(
                    lead,
                    "h",
                )
            )

            rows.append(
                {
                    "sid": args.sid,
                    "storm_name": storm_name,
                    "initialization_time": (
                        initialization_time
                    ),
                    "model": model,
                    "model_label": (
                        MODEL_LABELS.get(
                            model,
                            model,
                        )
                    ),
                    "lead_time_hours": lead,
                    "valid_time": str(
                        valid_time
                    ),
                    "ibtracs_latitude": obs_lat,
                    "ibtracs_longitude": obs_lon,
                    "minimum_mslp_hpa": minimum_mslp,
                    "environment_mslp_hpa": environment_mslp,
                    "pressure_deficit_hpa": pressure_deficit,
                    "pressure_center_error_km": (
                        pressure_center_error
                    ),
                    "maximum_wind10_ms": maximum_wind,
                    "maximum_zeta850_s1": maximum_zeta,
                    "maximum_zeta850_1e5_s1": (
                        maximum_zeta
                        * 1.0e5
                    ),
                    "mean_positive_zeta850_s1": (
                        mean_positive_zeta
                    ),
                    "mean_positive_zeta850_1e5_s1": (
                        mean_positive_zeta
                        * 1.0e5
                    ),
                    "zeta_center_error_km": (
                        zeta_center_error
                    ),
                    "pressure_zeta_separation_km": (
                        pressure_zeta_separation
                    ),
                    "organized_vortex": (
                        organized_vortex
                    ),
                }
            )

            print(
                f"+{lead:3d} h "
                f"Pmin={minimum_mslp:7.1f} "
                f"dP={pressure_deficit:4.1f} "
                f"V10={maximum_wind:5.1f} "
                f"Zmax={maximum_zeta * 1e5:5.1f} "
                f"Zmean+={mean_positive_zeta * 1e5:5.1f} "
                f"dPc={pressure_center_error:6.1f} "
                f"dZc={zeta_center_error:6.1f} "
                f"P-Z={pressure_zeta_separation:6.1f} "
                f"organized={organized_vortex}"
            )

        ds.close()

    results = pd.DataFrame(
        rows
    )

    if results.empty:
        raise RuntimeError(
            "No diagnostic records were generated."
        )

    results.to_csv(
        output_csv,
        index=False,
    )

    print()
    print("=" * 112)
    print("FIRST ORGANIZED VORTEX")
    print("=" * 112)

    for model in args.models:

        subset = results[
            (
                results["model"]
                == model
            )
            & (
                results[
                    "organized_vortex"
                ]
            )
        ]

        label = MODEL_LABELS.get(
            model,
            model,
        )

        if subset.empty:
            print()
            print(
                f"{label:10s}: "
                "not detected"
            )

            continue

        first = subset.sort_values(
            "lead_time_hours"
        ).iloc[0]

        print()
        print(
            f"{label:10s}: "
            f"+{int(first['lead_time_hours'])} h "
            f"({first['valid_time']})"
        )

    print()
    print("Saved:")
    print(output_csv)


if __name__ == "__main__":
    main()
