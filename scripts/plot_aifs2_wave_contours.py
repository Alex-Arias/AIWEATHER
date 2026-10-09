#!/usr/bin/env python3
"""Plot contour-based AIFS2 tropical-cyclone wave maps."""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

from wave_centers import load_wave_centers

import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.ticker import MultipleLocator


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--init", required=True)
    parser.add_argument("--storm", default="simon")
    parser.add_argument("--experiment-slug", default="invest_92e")
    parser.add_argument("--lead", type=int, default=48)
    args = parser.parse_args()

    init = pd.to_datetime(args.init, format="%Y%m%dT%H%M%S")
    valid = init + pd.Timedelta(hours=args.lead)

    forecast_path = (
        Path("outputs/aifs2")
        / args.init
        / "forecast.zarr"
    )

    track_path = (
        Path("results/operational")
        / f"{args.experiment_slug}_{args.init}"
        / "aifs2"
        / "aifs2_track.csv"
    )

    output_dir = (
        Path("results/waves")
        / f"aifs2_{args.experiment_slug}_{args.init[:8]}"
    )

    output_path = (
        output_dir
        / f"aifs2_{args.storm}_wave_{args.lead:03d}h_v5.png"
    )

    assert forecast_path.exists(), forecast_path
    assert track_path.is_file(), track_path

    # Protect previously generated figures.
    if output_path.exists():
        raise FileExistsError(
            f"Refusing to overwrite: {output_path}"
        )

    ds = xr.open_zarr(forecast_path)

    lead_values = (
        ds.lead_time.values
        .astype("timedelta64[h]")
        .astype(int)
    )

    matches = np.flatnonzero(lead_values == args.lead)
    if len(matches) != 1:
        raise ValueError(
            f"Lead {args.lead} h not uniquely available"
        )

    lead_index = int(matches[0])

    lat = np.asarray(ds.lat.values, dtype=float)
    lon = (
        np.asarray(ds.lon.values, dtype=float) + 180
    ) % 360 - 180

    lat_idx = np.flatnonzero(
        (lat >= 5) & (lat <= 35)
    )

    lon_idx = np.flatnonzero(
        (lon >= -140) & (lon <= -90)
    )

    lon_idx = lon_idx[np.argsort(lon[lon_idx])]

    lat_region = lat[lat_idx]
    lon_region = lon[lon_idx]

    def extract(name):
        return np.asarray(
            ds[name].isel(
                time=0,
                lead_time=lead_index,
                lat=lat_idx,
                lon=lon_idx,
            ).values,
            dtype=float,
        )

    swh = extract("swh")
    mwp = extract("mwp")
    sin_mwd = extract("sin_mwd")
    cos_mwd = extract("cos_mwd")
    u10 = extract("u10m")
    v10 = extract("v10m")

    wind = np.hypot(u10, v10)

    # ECMWF MWD is a wave-from bearing.
    # Reverse components to display propagation direction.
    norm = np.hypot(sin_mwd, cos_mwd)

    with np.errstate(invalid="ignore", divide="ignore"):
        wave_u = np.where(
            norm > 1e-6, -sin_mwd / norm, np.nan
        )
        wave_v = np.where(
            norm > 1e-6, -cos_mwd / norm, np.nan
        )

    # Reuse the validated AIWeather storm-center normalization.
    case = {
        "case_id": f"{args.experiment_slug}_{args.init}",
        "center_source": "native",
        "center_path": str(track_path),
    }

    track = load_wave_centers(case)

    required = [
        "lead_time_hours",
        "center_latitude",
        "center_longitude_plot",
    ]
    missing = [c for c in required if c not in track]
    if missing:
        raise ValueError(f"Missing normalized columns: {missing}")

    selected = track.loc[
        pd.to_numeric(
            track.lead_time_hours, errors="coerce"
        ) == args.lead
    ]

    if selected.empty:
        raise ValueError(
            f"No Native AIFS2 center at +{args.lead} h"
        )

    center = selected.iloc[0]
    tc_lat = float(center.center_latitude)
    tc_lon = float(center.center_longitude_plot)
    tc_lon = (tc_lon + 180) % 360 - 180

    projection = ccrs.PlateCarree()

    fig = plt.figure(figsize=(16, 9))

    # Two geographic panels per row, each with a reserved
    # colorbar column. The wave-direction colorbar is blank.
    gs = fig.add_gridspec(
        2, 4,
        width_ratios=[1, 0.035, 1, 0.035],
        height_ratios=[1, 1],
        wspace=0.12,
        hspace=0.18,
    )

    axes = [
        fig.add_subplot(gs[0, 0], projection=projection),
        fig.add_subplot(gs[0, 2], projection=projection),
        fig.add_subplot(gs[1, 0], projection=projection),
        fig.add_subplot(gs[1, 2], projection=projection),
    ]

    colorbar_axes = {
        axes[0]: fig.add_subplot(gs[0, 1]),
        axes[1]: fig.add_subplot(gs[0, 3]),
        axes[3]: fig.add_subplot(gs[1, 3]),
    }

    # Reserve the fourth colorbar slot for alignment.
    empty_colorbar_ax = fig.add_subplot(gs[1, 1])
    empty_colorbar_ax.axis("off")

    xx, yy = np.meshgrid(lon_region, lat_region)

    panels = [
        (
            axes[0], swh,
            np.arange(0, 10.5, 0.5),
            np.arange(0, 11, 1),
            "Significant Wave Height",
            "m",
            "YlOrRd",
        ),
        (
            axes[1], mwp,
            np.arange(2, 16, 2),
            np.arange(2, 16, 2),
            "Mean Wave Period",
            "s",
            "plasma",
        ),
        (
            axes[3], wind,
            np.arange(0, 40, 5),
            np.arange(0, 40, 5),
            "10-m Wind Speed",
            "m s$^{-1}$",
            "YlOrRd",
        ),
    ]

    for ax, field, levels, labels, title, unit, cmap in panels:
        cf = ax.contourf(
            xx, yy, field,
            levels=levels,
            cmap=cmap,
            extend="max",
            transform=projection,
        )

        cs = ax.contour(
            xx, yy, field,
            levels=labels,
            colors="black",
            linewidths=0.45,
            alpha=0.6,
            transform=projection,
        )

        ax.clabel(
            cs,
            inline=True,
            fontsize=7,
            fmt="%g",
        )

        cb = fig.colorbar(
            cf,
            cax=colorbar_axes[ax],
            orientation="vertical",
        )
        cb.set_label(unit)

        ax.set_title(title, fontsize=12)

    # Wave-direction panel: arrows only, no color shading.
    ax = axes[2]

    step = 12
    ax.quiver(
        xx[::step, ::step],
        yy[::step, ::step],
        wave_u[::step, ::step],
        wave_v[::step, ::step],
        color="black",
        scale=35,
        width=0.0025,
        transform=projection,
        zorder=1.5,
    )

    ax.set_title(
        "Mean Wave Propagation Direction",
        fontsize=12,
    )

    # Shared geographic context.
    for ax in axes:
        ax.set_extent(
            [-140, -90, 5, 35],
            crs=projection,
        )

        ax.add_feature(
            cfeature.LAND,
            facecolor="0.9",
            zorder=2,
        )

        ax.coastlines(
            resolution="110m",
            linewidth=0.8,
            zorder=3,
        )

        # Use standard Matplotlib ticks to avoid Cartopy
        # Gridliner/Shapely polygon-rendering failures.
        ax.set_xticks(
            np.arange(-140, -89, 10),
            crs=projection,
        )
        ax.set_yticks(
            np.arange(5, 36, 5),
            crs=projection,
        )

        ax.set_xticklabels(
            [f"{abs(x):.0f}°W" for x in np.arange(-140, -89, 10)],
            fontsize=8,
        )
        ax.set_yticklabels(
            [f"{y:.0f}°N" for y in np.arange(5, 36, 5)],
            fontsize=8,
        )

        ax.grid(
            True,
            linestyle="--",
            linewidth=0.4,
            alpha=0.45,
        )

        ax.plot(
            tc_lon,
            tc_lat,
            marker="*",
            markersize=13,
            color="red",
            markeredgecolor="black",
            transform=projection,
            zorder=10,
        )

        for radius in [300, 500, 800]:
            angle = np.linspace(0, 2 * np.pi, 361)

            ring_lat = (
                tc_lat
                + (radius / 111.0) * np.sin(angle)
            )

            ring_lon = (
                tc_lon
                + (
                    radius
                    / (
                        111.0
                        * max(
                            np.cos(np.deg2rad(tc_lat)),
                            0.1,
                        )
                    )
                ) * np.cos(angle)
            )

            ax.plot(
                ring_lon,
                ring_lat,
                linestyle="--",
                linewidth=0.7,
                color="black",
                alpha=0.5,
                transform=projection,
                zorder=5,
            )

    fig.suptitle(
        f"AIFS2 Wave Forecast — {args.storm.title()}\n"
        f"Initialized: {init:%Y-%m-%d %H:%M} UTC  |  "
        f"Valid: {valid:%Y-%m-%d %H:%M} UTC  |  "
        f"Lead: +{args.lead} h",
        fontsize=15,
        fontweight="bold",
    )

    fig.subplots_adjust(
        left=0.055,
        right=0.96,
        bottom=0.07,
        top=0.87,
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    fig.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)

    print("PASS — revised wave map created")
    print("Output:", output_path)
    print("Initialization:", init)
    print("Valid:", valid)
    print("Lead:", args.lead)
    print("Center:", tc_lat, tc_lon)


if __name__ == "__main__":
    main()
