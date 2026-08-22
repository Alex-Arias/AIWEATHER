from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from aiweather.forecast import open_forecast


# ============================================================
# Configuration
# ============================================================

FORECAST_PATH = Path(
    "outputs/aifs2/"
    "20260714T120000/"
    "forecast.zarr"
)

OUTPUT_DIR = Path(
    "results/waves/aifs2_elida_prototype"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

LEAD_TIME_HOURS = 72

LAT_MIN = 5.0
LAT_MAX = 35.0

LON_MIN = -130.0
LON_MAX = -90.0


# ============================================================
# Helpers
# ============================================================

def to_lon180(values):
    """
    Convert longitude from 0-360 to -180 to 180.
    """

    values = np.asarray(
        values,
        dtype=float,
    )

    return (
        (values + 180.0) % 360.0
        - 180.0
    )


# ============================================================
# Open forecast
# ============================================================

forecast = open_forecast(
    FORECAST_PATH
)

ds = forecast.dataset


# ============================================================
# Locate requested lead time
# ============================================================

lead_hours = (
    ds["lead_time"]
    .values
    .astype("timedelta64[h]")
    .astype(int)
)

matches = np.flatnonzero(
    lead_hours
    == LEAD_TIME_HOURS
)

if len(matches) == 0:
    raise ValueError(
        f"Lead time {LEAD_TIME_HOURS} h "
        "is not available."
    )

lead_index = int(
    matches[0]
)


# ============================================================
# Extract coordinates
# ============================================================

lat = np.asarray(
    ds["lat"].values
)

lon = to_lon180(
    ds["lon"].values
)


# ============================================================
# Longitude ordering
#
# Original AIFS2 longitude is 0-360.
# After conversion to -180/180 it is not monotonic,
# so sort it before plotting.
# ============================================================

lon_order = np.argsort(
    lon
)

lon_sorted = lon[
    lon_order
]


# ============================================================
# Regional mask
# ============================================================

lat_mask = (
    (lat >= LAT_MIN)
    & (lat <= LAT_MAX)
)

lon_mask = (
    (lon_sorted >= LON_MIN)
    & (lon_sorted <= LON_MAX)
)

lat_indices = np.flatnonzero(
    lat_mask
)

lon_indices_sorted = np.flatnonzero(
    lon_mask
)

lon_indices_original = lon_order[
    lon_indices_sorted
]

lat_region = lat[
    lat_indices
]

lon_region = lon_sorted[
    lon_indices_sorted
]


# ============================================================
# Extract fields
# ============================================================

def extract_region(
    variable_name,
):
    """
    Extract one regional field at the requested lead time.
    """

    data = (
        ds[variable_name]
        .isel(
            time=0,
            lead_time=lead_index,
            lat=lat_indices,
            lon=lon_indices_original,
        )
        .compute()
        .values
    )

    return np.asarray(
        data,
        dtype=float,
    )


swh = extract_region(
    "swh"
)

mwp = extract_region(
    "mwp"
)

cos_mwd = extract_region(
    "cos_mwd"
)

sin_mwd = extract_region(
    "sin_mwd"
)

u10 = extract_region(
    "u10m"
)

v10 = extract_region(
    "v10m"
)


# ============================================================
# Derived fields
# ============================================================

mwd = (
    np.degrees(
        np.arctan2(
            sin_mwd,
            cos_mwd,
        )
    )
    + 360.0
) % 360.0

wind_speed = np.sqrt(
    u10 ** 2
    + v10 ** 2
)


# ============================================================
# Basic diagnostics
# ============================================================

print("=" * 78)
print("AIFS2 WAVE PROTOTYPE")
print("=" * 78)

print(
    "Forecast:",
    FORECAST_PATH,
)

print(
    "Lead time:",
    LEAD_TIME_HOURS,
    "h",
)

print()

print(
    "SWH range:",
    f"{np.nanmin(swh):.2f}",
    "to",
    f"{np.nanmax(swh):.2f}",
    "m",
)

print(
    "MWP range:",
    f"{np.nanmin(mwp):.2f}",
    "to",
    f"{np.nanmax(mwp):.2f}",
    "s",
)

print(
    "MWD range:",
    f"{np.nanmin(mwd):.1f}",
    "to",
    f"{np.nanmax(mwd):.1f}",
    "deg",
)

print(
    "Wind-speed range:",
    f"{np.nanmin(wind_speed):.2f}",
    "to",
    f"{np.nanmax(wind_speed):.2f}",
    "m/s",
)


# ============================================================
# Plot
# ============================================================

fig, axes = plt.subplots(
    2,
    2,
    figsize=(14, 10),
)

ax = axes[0, 0]

pcm = ax.pcolormesh(
    lon_region,
    lat_region,
    swh,
    shading="auto",
)

fig.colorbar(
    pcm,
    ax=ax,
    label="Significant wave height (m)",
)

ax.set_title(
    "Significant Wave Height"
)


ax = axes[0, 1]

pcm = ax.pcolormesh(
    lon_region,
    lat_region,
    mwp,
    shading="auto",
)

fig.colorbar(
    pcm,
    ax=ax,
    label="Mean wave period (s)",
)

ax.set_title(
    "Mean Wave Period"
)


ax = axes[1, 0]

pcm = ax.pcolormesh(
    lon_region,
    lat_region,
    mwd,
    shading="auto",
    vmin=0.0,
    vmax=360.0,
)

fig.colorbar(
    pcm,
    ax=ax,
    label="Mean wave direction (°)",
)

ax.set_title(
    "Reconstructed Mean Wave Direction"
)


ax = axes[1, 1]

pcm = ax.pcolormesh(
    lon_region,
    lat_region,
    wind_speed,
    shading="auto",
)

fig.colorbar(
    pcm,
    ax=ax,
    label="10-m wind speed (m/s)",
)

ax.set_title(
    "10-m Wind Speed"
)


# ============================================================
# Common axes
# ============================================================

for ax in axes.flat:

    ax.set_xlim(
        LON_MIN,
        LON_MAX,
    )

    ax.set_ylim(
        LAT_MIN,
        LAT_MAX,
    )

    ax.set_xlabel(
        "Longitude (°)"
    )

    ax.set_ylabel(
        "Latitude (°)"
    )

    ax.grid(
        True,
        alpha=0.25,
    )


fig.suptitle(
    "AIFS2 Wave Forecast Prototype — Elida\n"
    f"Initialization: 2026-07-14 12 UTC | "
    f"Lead: +{LEAD_TIME_HOURS} h",
    fontsize=14,
    fontweight="bold",
)

fig.tight_layout(
    rect=[
        0,
        0,
        1,
        0.94,
    ]
)


# ============================================================
# Save
# ============================================================

png_path = (
    OUTPUT_DIR
    / (
        "aifs2_elida_wave_"
        f"{LEAD_TIME_HOURS:03d}h.png"
    )
)

pdf_path = (
    OUTPUT_DIR
    / (
        "aifs2_elida_wave_"
        f"{LEAD_TIME_HOURS:03d}h.pdf"
    )
)

fig.savefig(
    png_path,
    dpi=300,
    bbox_inches="tight",
)

fig.savefig(
    pdf_path,
    bbox_inches="tight",
)

plt.close(
    fig
)

print()
print("Saved:")
print(
    png_path
)
print(
    pdf_path
)