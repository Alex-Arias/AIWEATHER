"""
AIFS2 tropical-cyclone wave analysis.

Creates:
    1. Multi-panel wave maps at selected lead times.
    2. Storm-relative wave diagnostics around the configured TC center.
    3. CSV summary of wave conditions within 300, 500, and 800 km.
    4. Storm-relative SWH evolution figure.

The script reads an existing AIWeather AIFS2 forecast and a configured
storm-center source. It does not rerun AIFS2 or the tracker.
"""

from pathlib import Path

import argparse

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from aiweather.forecast import open_forecast
from wave_cases import get_case
from wave_centers import (
    load_wave_centers,
    longitude_near_reference,
)


# ============================================================
# Command-line configuration
# ============================================================

parser = argparse.ArgumentParser(
    description=(
        "AIFS2 tropical-cyclone wave analysis."
    )
)

parser.add_argument(
    "--storm",
    required=True,
    help=(
        "Configured storm key, or operational storm name "
        "when used together with --init."
    ),
)

parser.add_argument(
    "--init",
    default=None,
    help=(
        "Operational AIFS2 initialization in YYYYMMDDTHHMMSS "
        "format. When supplied, construct the wave case "
        "automatically instead of reading a configured case."
    ),
)


parser.add_argument(
    "--dry-run",
    action="store_true",
    help=(
        "Validate the wave case, forecast, storm track, "
        "and usable lead times without generating products."
    ),
)

args = parser.parse_args()


def build_operational_case(
    storm,
    init,
):
    """Construct one operational AIFS2 wave-analysis case."""

    storm_key = storm.lower()

    # Operational initialization must use the canonical
    # AIWeather timestamp format: YYYYMMDDTHHMMSS.
    try:
        initialization_time = pd.to_datetime(
            init,
            format="%Y%m%dT%H%M%S",
            errors="raise",
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "Invalid --init value "
            f"{init!r}. Expected YYYYMMDDTHHMMSS, "
            "for example 20260923T120000."
        ) from exc

    canonical_init = initialization_time.strftime(
        "%Y%m%dT%H%M%S"
    )

    if init != canonical_init:
        raise ValueError(
            "Invalid --init value "
            f"{init!r}. Expected canonical format "
            "YYYYMMDDTHHMMSS."
        )

    operational_dir = (
        Path("results/operational")
        / f"{storm_key}_{init}"
        / "aifs2"
    )

    center_path = (
        operational_dir
        / "aifs2_track.csv"
    )

    if not center_path.is_file():
        raise FileNotFoundError(
            "Operational AIFS2 track not found: "
            f"{center_path}"
        )

    forecast_dir = (
        Path("outputs/aifs2")
        / init
    )

    forecast_candidates = [
        forecast_dir / "forecast.zarr",
        forecast_dir / "forecast_azure.zarr",
    ]

    existing_forecasts = [
        candidate
        for candidate in forecast_candidates
        if candidate.exists()
    ]

    if not existing_forecasts:
        raise FileNotFoundError(
            "No AIFS2 forecast store found in "
            f"{forecast_dir}. Expected forecast.zarr "
            "or forecast_azure.zarr."
        )

    if len(existing_forecasts) > 1:
        raise RuntimeError(
            "Multiple AIFS2 forecast stores found: "
            + ", ".join(
                str(path)
                for path in existing_forecasts
            )
        )

    return {
        "storm_name": storm_key.title(),
        "init": init,
        "case_id": f"{storm_key}_{init}",
        "center_source": "operational",
        "center_path": str(center_path),
        "forecast_path": str(
            existing_forecasts[0]
        ),
        "wave_lead_times": [
            0,
            24,
            48,
            72,
            96,
            120,
            144,
            168,
            192,
            216,
            240,
        ],
        "lat_min": 5.0,
        "lat_max": 35.0,
        "lon_min": -140.0,
        "lon_max": -90.0,
        "swh_max": 12.0,
        "mwp_max": 18.0,
        "wind_max": 50.0,
    }


if args.init is None:
    CASE = get_case(
        args.storm
    )
else:
    CASE = build_operational_case(
        args.storm,
        args.init,
    )

CENTER_SOURCE = CASE.get(
    "center_source",
    "wuduan",
).lower()

CENTER_SOURCE_LABELS = {
    "wuduan": "WuDuan",
    "native": "Native",
    "ibtracs": "IBTrACS",
    "operational": "Operational",
}

CENTER_SOURCE_LABEL = CENTER_SOURCE_LABELS.get(
    CENTER_SOURCE,
    CENTER_SOURCE,
)

if args.init is None:
    STORM_KEY = args.storm.lower()
else:
    STORM_KEY = (
        f"{args.storm.lower()}_"
        f"{args.init[:8]}"
    )

STORM_NAME = CASE[
    "storm_name"
]

INIT = CASE[
    "init"
]

CASE_ID = CASE[
    "case_id"
]


# ============================================================
# Paths
# ============================================================

FORECAST_PATH = Path(
    CASE.get(
        "forecast_path",
        Path(
            "outputs/aifs2"
        ) / INIT / "forecast.zarr",
    )
)

OUTPUT_DIR = Path(
    "results/waves"
) / f"aifs2_{STORM_KEY}"


# ============================================================
# Analysis configuration
# ============================================================

LEAD_TIMES = CASE.get(
    "wave_lead_times",
    [
        0,
        24,
        48,
        72,
        96,
        120,
    ],
)

LAT_MIN = CASE.get(
    "lat_min",
    5.0,
)

LAT_MAX = CASE.get(
    "lat_max",
    35.0,
)

LON_MIN = CASE.get(
    "lon_min",
    -160.0,
)

LON_MAX = CASE.get(
    "lon_max",
    -90.0,
)

LONGITUDE_REFERENCE = CASE.get(
    "longitude_reference"
)

RADII_KM = [
    300.0,
    500.0,
    800.0,
]

QUIVER_STEP = 12


# ============================================================
# Fixed cross-storm plotting scales
# ============================================================

SWH_MIN = 0.0
SWH_MAX = CASE.get(
    "swh_max",
    7.0,
)

MWP_MIN = 2.0
MWP_MAX = CASE.get(
    "mwp_max",
    14.0,
)

MWD_MIN = 0.0
MWD_MAX = 360.0

WIND_MIN = 0.0
WIND_MAX = CASE.get(
    "wind_max",
    21.0,
)


OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


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


def haversine_distance_km(
    lat1,
    lon1,
    lat2,
    lon2,
):
    """
    Great-circle distance in kilometers.

    lat1/lon1 may be arrays.
    lat2/lon2 are the reference point.
    """

    radius_earth_km = 6371.0

    lat1 = np.radians(
        lat1
    )

    lon1 = np.radians(
        lon1
    )

    lat2 = np.radians(
        lat2
    )

    lon2 = np.radians(
        lon2
    )

    dlat = (
        lat1
        - lat2
    )

    dlon = (
        lon1
        - lon2
    )

    a = (
        np.sin(
            dlat / 2.0
        ) ** 2
        + np.cos(
            lat1
        )
        * np.cos(
            lat2
        )
        * np.sin(
            dlon / 2.0
        ) ** 2
    )

    return (
        2.0
        * radius_earth_km
        * np.arcsin(
            np.sqrt(
                a
            )
        )
    )


def reconstruct_mwd(
    sin_mwd,
    cos_mwd,
):
    """
    Reconstruct mean wave direction in degrees true.
    """

    return (
        np.degrees(
            np.arctan2(
                sin_mwd,
                cos_mwd,
            )
        )
        + 360.0
    ) % 360.0


def direction_orientation_vectors(
    direction_deg,
):
    """
    Convert directional bearing to orientation vectors.

    These vectors show directional orientation only.
    They are not yet interpreted as either wave-'from'
    or wave-propagation vectors.
    """

    theta = np.radians(
        direction_deg
    )

    u = np.sin(
        theta
    )

    v = np.cos(
        theta
    )

    return (
        u,
        v,
    )


# ============================================================
# Open forecast
# ============================================================

forecast = open_forecast(
    FORECAST_PATH
)

ds = forecast.dataset


# ============================================================
# Forecast lead times
# ============================================================

forecast_leads = (
    ds["lead_time"]
    .values
    .astype(
        "timedelta64[h]"
    )
    .astype(int)
)


# ============================================================
# Coordinate preparation
# ============================================================

lat = np.asarray(
    ds["lat"].values,
    dtype=float,
)

lon = to_lon180(
    ds["lon"].values
)

if LONGITUDE_REFERENCE is not None:

    lon = longitude_near_reference(
        lon,
        LONGITUDE_REFERENCE,
    )

lon_order = np.argsort(
    lon
)

lon_sorted = lon[
    lon_order
]


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


lon_grid, lat_grid = np.meshgrid(
    lon_region,
    lat_region,
)


# ============================================================
# Load storm-relative center positions
# ============================================================

track = load_wave_centers(
    CASE
)


# ============================================================
# Operational lead-time discovery
# ============================================================

if args.init is not None:
    track_leads = set(
        pd.to_numeric(
            track["lead_time_hours"],
            errors="coerce",
        )
        .dropna()
        .astype(int)
        .tolist()
    )

    forecast_lead_set = set(
        int(value)
        for value in forecast_leads
    )

    # Wave products are generated every 24 h.  Restrict them
    # to leads present in both the forecast and operational
    # storm track.
    LEAD_TIMES = [
        lead
        for lead in sorted(
            forecast_lead_set & track_leads
        )
        if lead % 24 == 0
    ]

    if not LEAD_TIMES:
        raise RuntimeError(
            "No common 24-hour wave-analysis leads found "
            "between the AIFS2 forecast and operational track."
        )

    print()
    print("OPERATIONAL WAVE LEAD DISCOVERY")
    print("=" * 60)
    print(
        "Forecast leads : "
        f"{min(forecast_lead_set)}–"
        f"{max(forecast_lead_set)} h "
        f"({len(forecast_lead_set)} available)"
    )
    print(
        "Track leads    : "
        f"{min(track_leads)}–"
        f"{max(track_leads)} h "
        f"({len(track_leads)} available)"
    )
    print(
        "Wave leads     : "
        + ", ".join(
            f"{lead}h"
            for lead in LEAD_TIMES
        )
    )


if args.dry_run:
    print()
    print("DRY RUN")
    print("=" * 60)
    print(f"Storm          : {STORM_NAME}")
    print(f"Initialization : {CASE['init']}")
    print(f"Center source  : {CENTER_SOURCE}")
    print(f"Center path    : {CASE['center_path']}")
    print(f"Forecast path  : {FORECAST_PATH}")
    print(
        "Wave leads     : "
        + ", ".join(
            f"{lead}h"
            for lead in LEAD_TIMES
        )
    )
    print()
    print(
        "PASS — operational wave case is ready "
        "for analysis."
    )
    raise SystemExit(0)


# ============================================================
# Forecast extraction helper
# ============================================================

def get_lead_index(
    lead_hours,
):
    """
    Return dataset index for one lead time.
    """

    matches = np.flatnonzero(
        forecast_leads
        == lead_hours
    )

    if len(matches) == 0:
        raise ValueError(
            f"Lead {lead_hours} h "
            "not available in forecast."
        )

    return int(
        matches[0]
    )


def extract_region(
    variable_name,
    lead_hours,
):
    """
    Extract regional forecast variable.
    """

    lead_index = get_lead_index(
        lead_hours
    )

    values = (
        ds[
            variable_name
        ]
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
        values,
        dtype=float,
    )


def get_tc_center(
    lead_hours,
):
    """
    Return configured TC center nearest to exact lead time.
    """

    subset = track[
        track[
            "lead_time_hours"
        ]
        == lead_hours
    ]

    if subset.empty:
        return None

    row = subset.iloc[0]

    return (
        float(
            row[
                "center_latitude"
            ]
        ),
        float(
            row[
                "center_longitude_plot"
            ]
        ),
    )


# ============================================================
# Plot maps
# ============================================================

map_paths = []

for lead_hours in LEAD_TIMES:

    print()
    print(
        "=" * 78
    )

    print(
        f"{STORM_NAME}: +{lead_hours} h"
    )

    print(
        "=" * 78
    )

    swh = extract_region(
        "swh",
        lead_hours,
    )

    mwp = extract_region(
        "mwp",
        lead_hours,
    )

    sin_mwd = extract_region(
        "sin_mwd",
        lead_hours,
    )

    cos_mwd = extract_region(
        "cos_mwd",
        lead_hours,
    )

    u10 = extract_region(
        "u10m",
        lead_hours,
    )

    v10 = extract_region(
        "v10m",
        lead_hours,
    )


    mwd = reconstruct_mwd(
        sin_mwd,
        cos_mwd,
    )

    wind_speed = np.sqrt(
        u10 ** 2
        + v10 ** 2
    )

    wave_u, wave_v = (
        direction_orientation_vectors(
            mwd
        )
    )


    tc_center = get_tc_center(
        lead_hours
    )


    print(
        "SWH:",
        f"{np.nanmin(swh):.2f}",
        "to",
        f"{np.nanmax(swh):.2f}",
        "m",
    )

    print(
        "MWP:",
        f"{np.nanmin(mwp):.2f}",
        "to",
        f"{np.nanmax(mwp):.2f}",
        "s",
    )

    print(
        "Wind:",
        f"{np.nanmin(wind_speed):.2f}",
        "to",
        f"{np.nanmax(wind_speed):.2f}",
        "m/s",
    )


    fig, axes = plt.subplots(
        2,
        2,
        figsize=(
            14,
            10,
        ),
    )


    # --------------------------------------------------------
    # SWH
    # --------------------------------------------------------

    ax = axes[
        0,
        0,
    ]

    pcm = ax.pcolormesh(
        lon_region,
        lat_region,
        swh,
        shading="auto",
        vmin=SWH_MIN,
        vmax=SWH_MAX,
    )

    fig.colorbar(
        pcm,
        ax=ax,
        label=(
            "Significant wave height (m)"
        ),
    )

    ax.set_title(
        "Significant Wave Height"
    )


    # --------------------------------------------------------
    # Mean wave period
    # --------------------------------------------------------

    ax = axes[
        0,
        1,
    ]

    pcm = ax.pcolormesh(
        lon_region,
        lat_region,
        mwp,
        shading="auto",
        vmin=MWP_MIN,
        vmax=MWP_MAX,
    )

    fig.colorbar(
        pcm,
        ax=ax,
        label="Mean wave period (s)",
    )

    ax.set_title(
        "Mean Wave Period"
    )


    # --------------------------------------------------------
    # Direction
    # --------------------------------------------------------

    ax = axes[
        1,
        0,
    ]

    pcm = ax.pcolormesh(
        lon_region,
        lat_region,
        mwd,
        shading="auto",
        cmap="twilight",
        vmin=MWD_MIN,
        vmax=MWD_MAX,
    )

    fig.colorbar(
        pcm,
        ax=ax,
        label=(
            "Mean wave direction (°)"
        ),
        ticks=[
            0,
            90,
            180,
            270,
            360,
        ],
    )


    step = QUIVER_STEP

    ax.quiver(
        lon_grid[
            ::step,
            ::step,
        ],
        lat_grid[
            ::step,
            ::step,
        ],
        wave_u[
            ::step,
            ::step,
        ],
        wave_v[
            ::step,
            ::step,
        ],
        scale=35,
        width=0.002,
        alpha=0.7,
    )

    ax.set_title(
        "Mean Wave Direction"
    )


    # --------------------------------------------------------
    # Wind
    # --------------------------------------------------------

    ax = axes[
        1,
        1,
    ]

    pcm = ax.pcolormesh(
        lon_region,
        lat_region,
        wind_speed,
        shading="auto",
        vmin=WIND_MIN,
        vmax=WIND_MAX,
    )

    fig.colorbar(
        pcm,
        ax=ax,
        label=(
            "10-m wind speed (m/s)"
        ),
    )

    ax.set_title(
        "10-m Wind Speed"
    )


    # ========================================================
    # TC center and radius overlays
    # ========================================================

    if tc_center is not None:

        tc_lat, tc_lon = (
            tc_center
        )

        for ax in axes.flat:

            ax.scatter(
                tc_lon,
                tc_lat,
                marker="*",
                s=150,
                edgecolor="black",
                linewidth=0.8,
                zorder=10,
                label=f"{CENTER_SOURCE_LABEL} TC center",
            )


            # Approximate geographic circles.
            # Good enough for visualization only.
            for radius_km in RADII_KM:

                angular_radius = (
                    radius_km
                    / 111.0
                )

                theta = np.linspace(
                    0.0,
                    2.0 * np.pi,
                    200,
                )

                circle_lat = (
                    tc_lat
                    + angular_radius
                    * np.sin(
                        theta
                    )
                )

                longitude_scale = (
                    np.cos(
                        np.radians(
                            tc_lat
                        )
                    )
                )

                circle_lon = (
                    tc_lon
                    + (
                        angular_radius
                        / longitude_scale
                    )
                    * np.cos(
                        theta
                    )
                )

                ax.plot(
                    circle_lon,
                    circle_lat,
                    linewidth=0.8,
                    linestyle="--",
                    alpha=0.5,
                )


    # ========================================================
    # Common axes
    # ========================================================

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
        f"AIFS2 Wave Forecast — {STORM_NAME}\n"
        f"Lead: +{lead_hours} h",
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


    png_path = (
        OUTPUT_DIR
        / (
            f"aifs2_{STORM_NAME.lower()}_"
            f"wave_{lead_hours:03d}h.png"
        )
    )

    pdf_path = (
        OUTPUT_DIR
        / (
            f"aifs2_{STORM_NAME.lower()}_"
            f"wave_{lead_hours:03d}h.pdf"
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

    map_paths.append(
        png_path
    )


# ============================================================
# Storm-relative wave diagnostics
# ============================================================

diagnostic_rows = []


for _, row in track.iterrows():

    lead_hours = int(
        row[
            "lead_time_hours"
        ]
    )

    if lead_hours not in forecast_leads:
        continue


    tc_lat = float(
        row[
            "center_latitude"
        ]
    )

    tc_lon = float(
        row[
            "center_longitude_plot"
        ]
    )


    swh = extract_region(
        "swh",
        lead_hours,
    )

    mwp = extract_region(
        "mwp",
        lead_hours,
    )

    u10 = extract_region(
        "u10m",
        lead_hours,
    )

    v10 = extract_region(
        "v10m",
        lead_hours,
    )

    wind_speed = np.sqrt(
        u10 ** 2
        + v10 ** 2
    )


    distance_km = (
        haversine_distance_km(
            lat_grid,
            lon_grid,
            tc_lat,
            tc_lon,
        )
    )

    # --------------------------------------------------------
    # Location of regional SWH maximum
    # --------------------------------------------------------

    regional_swh_index = np.unravel_index(
        np.nanargmax(
            swh
        ),
        swh.shape,
    )

    regional_max_swh_latitude = float(
        lat_grid[
            regional_swh_index
        ]
    )

    regional_max_swh_longitude = float(
        lon_grid[
            regional_swh_index
        ]
    )

    regional_max_swh_distance_from_tc_km = float(
        haversine_distance_km(
            regional_max_swh_latitude,
            regional_max_swh_longitude,
            tc_lat,
            tc_lon,
        )
    )

    # --------------------------------------------------------
    # Location of regional wind-speed maximum
    # --------------------------------------------------------

    regional_wind_index = np.unravel_index(
        np.nanargmax(
            wind_speed
        ),
        wind_speed.shape,
    )

    regional_max_wind_latitude = float(
        lat_grid[
            regional_wind_index
        ]
    )

    regional_max_wind_longitude = float(
        lon_grid[
            regional_wind_index
        ]
    )

    regional_max_wind_distance_from_tc_km = float(
        haversine_distance_km(
            regional_max_wind_latitude,
            regional_max_wind_longitude,
            tc_lat,
            tc_lon,
        )
    )

    result = {
        "case_id":
            CASE_ID,

        "storm":
            STORM_NAME,

        "lead_time_hours":
            lead_hours,

        "tc_latitude":
            tc_lat,

        "tc_longitude":
            tc_lon,

        "regional_max_swh_m":
            np.nanmax(
                swh
            ),

        "regional_max_swh_latitude":
            regional_max_swh_latitude,

        "regional_max_swh_longitude":
            regional_max_swh_longitude,

        "regional_max_swh_distance_from_tc_km":
            regional_max_swh_distance_from_tc_km,

        "regional_max_wind_ms":
            np.nanmax(
                wind_speed
            ),

        "regional_max_wind_latitude":
            regional_max_wind_latitude,

        "regional_max_wind_longitude":
            regional_max_wind_longitude,

        "regional_max_wind_distance_from_tc_km":
            regional_max_wind_distance_from_tc_km,

    }


    for radius_km in RADII_KM:

        mask = (
            distance_km
            <= radius_km
        )

        radius_label = int(
            radius_km
        )


        result[
            f"max_swh_{radius_label}km_m"
        ] = np.nanmax(
            np.where(
                mask,
                swh,
                np.nan,
            )
        )


        result[
            f"mean_swh_{radius_label}km_m"
        ] = np.nanmean(
            np.where(
                mask,
                swh,
                np.nan,
            )
        )


        result[
            f"max_wind_{radius_label}km_ms"
        ] = np.nanmax(
            np.where(
                mask,
                wind_speed,
                np.nan,
            )
        )


        result[
            f"mean_mwp_{radius_label}km_s"
        ] = np.nanmean(
            np.where(
                mask,
                mwp,
                np.nan,
            )
        )


    diagnostic_rows.append(
        result
    )


diagnostics = pd.DataFrame(
    diagnostic_rows
)

diagnostics = diagnostics.sort_values(
    "lead_time_hours"
)


csv_path = (
    OUTPUT_DIR
    / (
        f"aifs2_{STORM_NAME.lower()}_"
        "wave_diagnostics.csv"
    )
)

diagnostics.to_csv(
    csv_path,
    index=False,
)


# ============================================================
# Storm-relative SWH evolution figure
# ============================================================

fig, ax = plt.subplots(
    figsize=(
        10,
        6,
    )
)


for radius_km in RADII_KM:

    radius_label = int(
        radius_km
    )

    ax.plot(
        diagnostics[
            "lead_time_hours"
        ],
        diagnostics[
            f"max_swh_{radius_label}km_m"
        ],
        marker="o",
        linewidth=2,
        label=(
            f"Maximum SWH "
            f"within {radius_label} km"
        ),
    )


ax.set_xlabel(
    "Forecast lead time (h)"
)

ax.set_ylabel(
    "Maximum significant wave height (m)"
)

ax.set_title(
    f"AIFS2 Storm-Relative Wave Evolution — {STORM_NAME}"
)

ax.grid(
    True,
    alpha=0.3,
)

ax.legend()


fig.tight_layout()


swh_png = (
    OUTPUT_DIR
    / (
        f"aifs2_{STORM_NAME.lower()}_"
        "storm_relative_swh.png"
    )
)

swh_pdf = (
    OUTPUT_DIR
    / (
        f"aifs2_{STORM_NAME.lower()}_"
        "storm_relative_swh.pdf"
    )
)


fig.savefig(
    swh_png,
    dpi=300,
    bbox_inches="tight",
)

fig.savefig(
    swh_pdf,
    bbox_inches="tight",
)

plt.close(
    fig
)


# ============================================================
# Console summary
# ============================================================

print()
print(
    "=" * 78
)

print(
    "AIFS2 WAVE ANALYSIS COMPLETE"
)

print(
    "=" * 78
)

print()

print(
    "Forecast:",
    FORECAST_PATH,
)

print(
    "Storm:",
    STORM_NAME,
)

print(
    "Storm center source:",
    CENTER_SOURCE_LABEL,
)

print()

print(
    "Wave diagnostics:"
)

print(
    csv_path
)

print()

print(
    diagnostics.to_string(
        index=False,
        float_format=lambda value:
            f"{value:.2f}",
    )
)

print()

print(
    "Storm-relative SWH figure:"
)

print(
    swh_png
)

print()

print(
    "Wave maps:"
)

for path in map_paths:
    print(
        path
    )