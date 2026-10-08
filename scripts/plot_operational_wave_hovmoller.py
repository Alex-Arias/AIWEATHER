from pathlib import Path
import argparse

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from aiweather.forecast import open_forecast


# ============================================================
# Command-line configuration
# ============================================================

parser = argparse.ArgumentParser(
    description=(
        "Plot storm-relative Hovmoller diagnostics for one "
        "operational AIFS2 tropical-cyclone forecast."
    )
)

parser.add_argument(
    "--storm",
    required=True,
    help="Operational tropical-cyclone name.",
)

parser.add_argument(
    "--init",
    required=True,
    help="Initialization in YYYYMMDDTHHMMSS format.",
)

parser.add_argument(
    "--experiment-slug",
    default=None,
    help=(
        "Persistent operational experiment directory slug. "
        "Defaults to the normalized storm name."
    ),
)

args = parser.parse_args()


def storm_slug(name):
    """Return the canonical filesystem slug for a storm name."""
    return (
        name.strip()
        .lower()
        .replace(" ", "_")
    )


STORM_NAME = args.storm.strip().replace("_", " ")
STORM_KEY = storm_slug(args.storm)

EXPERIMENT_KEY = (
    STORM_KEY
    if args.experiment_slug is None
    else args.experiment_slug
)

if not EXPERIMENT_KEY or not all(
    character in "abcdefghijklmnopqrstuvwxyz0123456789_"
    for character in EXPERIMENT_KEY
):
    raise ValueError(
        "Experiment slug must contain only lowercase "
        "letters, digits, and underscores."
    )

INIT = args.init.strip()

CASE_ID = (
    f"operational_{EXPERIMENT_KEY}_{INIT.lower()}"
)

CENTER_SOURCE = "native"
CENTER_SOURCE_LABEL = "Native"

# ============================================================
# Paths
# ============================================================

FORECAST_PATH = (
    Path("outputs/aifs2")
    / INIT
    / "forecast.zarr"
)

TRACK_PATH = (
    Path("results/operational")
    / f"{EXPERIMENT_KEY}_{INIT}"
    / "aifs2"
    / "aifs2_track.csv"
)

OUTPUT_DIR = (
    Path("results/waves")
    / f"aifs2_{EXPERIMENT_KEY}_{INIT[:8]}"
)

LAT_MIN = 5.0
LAT_MAX = 35.0
LON_MIN = -160.0
LON_MAX = -90.0

LONGITUDE_REFERENCE = None

RADIAL_BIN_KM = 50.0
MAX_RADIUS_KM = 800.0

REFERENCE_LEADS = [
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
]

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

if not FORECAST_PATH.exists():
    raise FileNotFoundError(
        f"Forecast not found: {FORECAST_PATH}"
    )

if not TRACK_PATH.is_file():
    raise FileNotFoundError(
        f"Native track not found: {TRACK_PATH}"
    )

# ============================================================
# Helpers
# ============================================================

def to_lon180(values):

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

    earth_radius_km = 6371.0

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
        * earth_radius_km
        * np.arcsin(
            np.sqrt(
                a
            )
        )
    )


# ============================================================
# Open forecast
# ============================================================

forecast = open_forecast(
    FORECAST_PATH
)

ds = forecast.dataset


forecast_leads = (
    ds["lead_time"]
    .values
    .astype(
        "timedelta64[h]"
    )
    .astype(int)
)


# ============================================================
# Coordinates
# ============================================================

lat = np.asarray(
    ds["lat"].values,
    dtype=float,
)

lon = to_lon180(
    ds["lon"].values
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

track = pd.read_csv(
    TRACK_PATH
)

required_track_columns = {
    "lead_time_hours",
    "latitude",
    "longitude",
}

missing_track_columns = (
    required_track_columns
    - set(track.columns)
)

if missing_track_columns:
    raise ValueError(
        "Operational native track is missing columns: "
        + ", ".join(
            sorted(missing_track_columns)
        )
    )

track = (
    track[
        [
            "lead_time_hours",
            "latitude",
            "longitude",
        ]
    ]
    .dropna()
    .sort_values("lead_time_hours")
    .reset_index(drop=True)
)


# ============================================================
# Extraction helper
# ============================================================

def extract_region(
    variable_name,
    lead_hours,
):

    matches = np.flatnonzero(
        forecast_leads
        == lead_hours
    )

    if len(matches) == 0:
        return None

    lead_index = int(
        matches[0]
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


# ============================================================
# Radial bins
# ============================================================

radial_edges = np.arange(
    0.0,
    MAX_RADIUS_KM
    + RADIAL_BIN_KM,
    RADIAL_BIN_KM,
)

radial_centers = (
    radial_edges[:-1]
    + radial_edges[1:]
) / 2.0


lead_values = []

swh_radial_profiles = []

wind_radial_profiles = []


# ============================================================
# Loop over tracked lead times
# ============================================================

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
            "latitude"
        ]
    )

    tc_lon = float(
        (
            float(
                row[
                    "longitude"
                ]
            )
            + 180.0
        )
        % 360.0
        - 180.0
    )


    swh = extract_region(
        "swh",
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


    swh_profile = np.full(
        radial_centers.shape,
        np.nan,
        dtype=float,
    )

    wind_profile = np.full(
        radial_centers.shape,
        np.nan,
        dtype=float,
    )


    for index in range(
        len(
            radial_centers
        )
    ):

        r0 = radial_edges[
            index
        ]

        r1 = radial_edges[
            index + 1
        ]


        mask = (
            (distance_km >= r0)
            & (distance_km < r1)
        )


        if not np.any(
            mask
        ):
            continue


        swh_profile[
            index
        ] = np.nanmean(
            swh[
                mask
            ]
        )


        wind_profile[
            index
        ] = np.nanmean(
            wind_speed[
                mask
            ]
        )


    lead_values.append(
        lead_hours
    )

    swh_radial_profiles.append(
        swh_profile
    )

    wind_radial_profiles.append(
        wind_profile
    )


# ============================================================
# Convert arrays
# ============================================================

lead_values = np.asarray(
    lead_values,
    dtype=float,
)

swh_matrix = np.asarray(
    swh_radial_profiles
).T

wind_matrix = np.asarray(
    wind_radial_profiles
).T


# ============================================================
# Sort by lead time
# ============================================================

order = np.argsort(
    lead_values
)

lead_values = lead_values[
    order
]

swh_matrix = swh_matrix[
    :,
    order,
]

wind_matrix = wind_matrix[
    :,
    order,
]


# ============================================================
# Radius of maximum radial-mean SWH
# ============================================================

max_swh_radius = np.full(
    lead_values.shape,
    np.nan,
    dtype=float,
)

max_swh_value = np.full(
    lead_values.shape,
    np.nan,
    dtype=float,
)

for time_index in range(
    len(lead_values)
):

    profile = swh_matrix[
        :,
        time_index,
    ]

    if np.all(
        np.isnan(profile)
    ):
        continue

    radial_index = int(
        np.nanargmax(
            profile
        )
    )

    max_swh_radius[
        time_index
    ] = radial_centers[
        radial_index
    ]

    max_swh_value[
        time_index
    ] = profile[
        radial_index
    ]


# ============================================================
# Plot
# ============================================================

fig, ax = plt.subplots(
    figsize=(
        12,
        7,
    )
)


pcm = ax.pcolormesh(
    lead_values,
    radial_centers,
    swh_matrix,
    shading="auto",
    vmin=0.0,
    vmax=10.0,
)


colorbar = fig.colorbar(
    pcm,
    ax=ax,
)

colorbar.set_label(
    "Mean significant wave height (m)"
)


# ============================================================
# Wind contours
# ============================================================

contour_levels = [
    10,
    12,
    14,
    16,
]

cs = ax.contour(
    lead_values,
    radial_centers,
    wind_matrix,
    levels=contour_levels,
    linewidths=1.0,
)


ax.clabel(
    cs,
    inline=True,
    fontsize=8,
    fmt="%g m/s",
)


# ============================================================
# Radius of maximum radial-mean SWH
# ============================================================

ax.plot(
    lead_values,
    max_swh_radius,
    color="black",
    linewidth=2.0,
    marker="o",
    markersize=3.5,
    label="Radius of maximum radial-mean SWH",
    zorder=5,
)

ax.legend(
    loc="upper right",
    fontsize=8,
)


# ============================================================
# Reference lead times
# ============================================================

for lead in REFERENCE_LEADS:

    ax.axvline(
        lead,
        linestyle=":",
        linewidth=0.9,
        alpha=0.6,
    )


# ============================================================
# Labels
# ============================================================

ax.set_xlabel(
    "Forecast lead time (h)"
)

ax.set_ylabel(
    f"Radius from {CENTER_SOURCE_LABEL} TC center (km)"
)

ax.set_title(
    f"AIFS2 Storm-Relative Wave Evolution — {STORM_NAME}\n"
    "Radial mean SWH with 10-m wind contours"
)

ax.set_ylim(
    0.0,
    MAX_RADIUS_KM,
)

ax.grid(
    True,
    alpha=0.15,
)


fig.tight_layout()


# ============================================================
# Save
# ============================================================

png_path = (
    OUTPUT_DIR
    / (
        f"{STORM_KEY}_{INIT}_wave_hovmoller.png"
    )
)

pdf_path = (
    OUTPUT_DIR
    / (
        f"{STORM_KEY}_{INIT}_wave_hovmoller.pdf"
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


# ============================================================
# Export radial table
# ============================================================

rows = []

for time_index, lead_hours in enumerate(
    lead_values
):

    for radius_index, radius_km in enumerate(
        radial_centers
    ):

        rows.append(
            {
                "case_id":
                    CASE_ID,

                "storm":
                    STORM_NAME,

                "lead_time_hours":
                    int(
                        lead_hours
                    ),

                "radius_km":
                    float(
                        radius_km
                    ),

                "mean_swh_m":
                    swh_matrix[
                        radius_index,
                        time_index,
                    ],

                "mean_wind_ms":
                    wind_matrix[
                        radius_index,
                        time_index,
                    ],

                "is_max_swh_radius": (
                    np.isclose(
                        radius_km,
                        max_swh_radius[
                            time_index
                        ],
                        equal_nan=False,
                    )
                ),

                "max_radial_mean_swh_m":
                    max_swh_value[
                        time_index
                    ],
            }
        )


radial_table = pd.DataFrame(
    rows
)


csv_path = (
    OUTPUT_DIR
    / (
        f"{STORM_KEY}_{INIT}_wave_radial_profiles.csv"
    )
)


radial_table.to_csv(
    csv_path,
    index=False,
)


# ============================================================
# Summary
# ============================================================

print("=" * 78)
print(
    f"AIFS2 {STORM_NAME.upper()} "
    "WAVE HOVMOLLER"
)
print("=" * 78)

print()
print(
    "Lead times:",
    lead_values.astype(int)
)

print()
print(
    "Radial centers:"
)

print(
    radial_centers
)

print()
print(
    "Saved:"
)

print(
    png_path
)

print(
    pdf_path
)

print(
    csv_path
)

print()
print(
    "SWH matrix range:",
    f"{np.nanmin(swh_matrix):.2f}",
    "to",
    f"{np.nanmax(swh_matrix):.2f}",
    "m",
)

print(
    "Wind matrix range:",
    f"{np.nanmin(wind_matrix):.2f}",
    "to",
    f"{np.nanmax(wind_matrix):.2f}",
    "m/s",
)

print()
print(
    "Radius of maximum radial-mean SWH:"
)

for (
    lead_hours,
    radius_km,
    value_m,
) in zip(
    lead_values,
    max_swh_radius,
    max_swh_value,
):

    print(
        f"+{int(lead_hours):3d} h  "
        f"r={radius_km:5.0f} km  "
        f"SWH={value_m:4.2f} m"
    )