"""
General AIWeather tropical-cyclone track variability diagnostic.

This tool separates three related quantities:

1. Historical variability
   All available forecast cycles.

2. Recent variability
   The N most recent forecast cycles.

3. Current multimodel spread
   The latest forecast cycle only.

Forecasts are aligned by VALID TIME.

The resulting spread statistics describe disagreement among AIWeather
forecast tracks. They are NOT calibrated tropical-cyclone probabilities
and should not be interpreted as an official forecast cone.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


EARTH_RADIUS_KM = 6371.0088


# ============================================================
# CLI
# ============================================================

def parse_args():

    parser = argparse.ArgumentParser(
        description=(
            "Compute historical, recent-cycle, and latest-cycle "
            "AIWeather tropical-cyclone track variability."
        )
    )

    parser.add_argument(
        "--members",
        required=True,
        type=Path,
        help=(
            "CSV containing aligned track members. Required columns: "
            "valid_time, cycle, model, latitude, longitude."
        ),
    )

    parser.add_argument(
        "--storm",
        required=True,
        help="Storm name used in output filenames.",
    )

    parser.add_argument(
        "--tracker",
        default="wuduan",
        help="Tracker name used in output filenames. Default: wuduan.",
    )

    parser.add_argument(
        "--recent-cycles",
        type=int,
        default=3,
        help="Number of newest cycles used for recent variability. Default: 3.",
    )

    parser.add_argument(
        "--latest-cycle",
        type=int,
        default=None,
        help=(
            "Optional cycle override for the current forecast. "
            "Default: newest cycle in the members file."
        ),
    )

    parser.add_argument(
        "--minimum-models",
        type=int,
        default=3,
        help=(
            "Minimum number of distinct models required for a valid "
            "historical/recent spread estimate. Default: 3."
        ),
    )

    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Output directory.",
    )

    parser.add_argument(
        "--plots",
        action="store_true",
        help=(
            "Generate the three-panel track-variability map."
        ),
    )

    parser.add_argument(
        "--extent",
        nargs=4,
        type=float,
        metavar=("LON_MIN", "LON_MAX", "LAT_MIN", "LAT_MAX"),
        default=None,
        help=(
            "Optional map extent. If omitted, the domain is "
            "computed automatically from all track members."
        ),
    )

    return parser.parse_args()


# ============================================================
# VALIDATION
# ============================================================

def validate_args(args):

    if args.recent_cycles < 1:
        raise ValueError(
            "--recent-cycles must be >= 1."
        )

    if args.minimum_models < 1:
        raise ValueError(
            "--minimum-models must be >= 1."
        )

    if not args.members.exists():
        raise FileNotFoundError(
            f"Members file does not exist: {args.members}"
        )


def validate_members(df):

    required = {
        "valid_time",
        "cycle",
        "model",
        "latitude",
        "longitude",
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            "Members CSV is missing required columns: "
            + ", ".join(sorted(missing))
        )

    if df.empty:
        raise ValueError(
            "Members CSV contains no rows."
        )

    if df[
        ["latitude", "longitude"]
    ].isna().any().any():
        raise ValueError(
            "Members CSV contains missing latitude/longitude values."
        )


# ============================================================
# LONGITUDE NORMALIZATION
# ============================================================

def normalize_longitude(lon):
    """Normalize longitude values to the [-180, 180) convention."""

    values = np.asarray(
        lon,
        dtype=float,
    )

    return (
        (values + 180.0) % 360.0
    ) - 180.0


# ============================================================
# SPHERICAL GEOMETRY
# ============================================================

def spherical_mean(lat_deg, lon_deg):

    lat = np.radians(
        np.asarray(lat_deg, dtype=float)
    )

    lon = np.radians(
        np.asarray(lon_deg, dtype=float)
    )

    x = np.cos(lat) * np.cos(lon)
    y = np.cos(lat) * np.sin(lon)
    z = np.sin(lat)

    xbar = np.mean(x)
    ybar = np.mean(y)
    zbar = np.mean(z)

    norm = np.sqrt(
        xbar * xbar
        + ybar * ybar
        + zbar * zbar
    )

    if norm == 0.0:
        raise ValueError(
            "Undefined spherical mean."
        )

    xbar /= norm
    ybar /= norm
    zbar /= norm

    lat_mean = np.degrees(
        np.arctan2(
            zbar,
            np.sqrt(
                xbar * xbar
                + ybar * ybar
            ),
        )
    )

    lon_mean = np.degrees(
        np.arctan2(
            ybar,
            xbar,
        )
    )

    return lat_mean, lon_mean


def haversine_km(
    lat1,
    lon1,
    lat2,
    lon2,
):

    lat1 = np.radians(float(lat1))
    lon1 = np.radians(float(lon1))

    lat2 = np.radians(
        np.asarray(lat2, dtype=float)
    )

    lon2 = np.radians(
        np.asarray(lon2, dtype=float)
    )

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = (
        np.sin(dlat / 2.0) ** 2
        + np.cos(lat1)
        * np.cos(lat2)
        * np.sin(dlon / 2.0) ** 2
    )

    a = np.clip(
        a,
        0.0,
        1.0,
    )

    return (
        2.0
        * EARTH_RADIUS_KM
        * np.arcsin(np.sqrt(a))
    )


# ============================================================
# VARIABILITY CALCULATION
# ============================================================

def compute_variability(df):

    rows = []

    for valid_time, g in df.groupby(
        "valid_time",
        sort=True,
    ):

        lat0, lon0 = spherical_mean(
            g["latitude"].values,
            g["longitude"].values,
        )

        distances = haversine_km(
            lat0,
            lon0,
            g["latitude"].values,
            g["longitude"].values,
        )

        row = {
            "valid_time": valid_time,

            "consensus_latitude": lat0,
            "consensus_longitude": lon0,

            "n_members": len(g),
            "n_cycles": g["cycle"].nunique(),
            "n_models": g["model"].nunique(),

            "r67_km": np.quantile(
                distances,
                0.67,
            ),

            "r90_km": np.quantile(
                distances,
                0.90,
            ),

            "mean_distance_km":
                np.mean(distances),

            "median_distance_km":
                np.median(distances),

            "max_distance_km":
                np.max(distances),
        }

        if "lead_time_hours" in g.columns:
            row["min_lead_h"] = (
                g["lead_time_hours"].min()
            )

            row["max_lead_h"] = (
                g["lead_time_hours"].max()
            )

        rows.append(row)

    return (
        pd.DataFrame(rows)
        .sort_values("valid_time")
        .reset_index(drop=True)
    )


# ============================================================
# OPTIONAL PLOTTING
# ============================================================

def destination_point(
    lat,
    lon,
    bearing_deg,
    distance_km,
):

    lat1 = np.radians(lat)
    lon1 = np.radians(lon)
    brng = np.radians(bearing_deg)

    delta = distance_km / EARTH_RADIUS_KM

    lat2 = np.arcsin(
        np.sin(lat1) * np.cos(delta)
        + np.cos(lat1)
        * np.sin(delta)
        * np.cos(brng)
    )

    lon2 = lon1 + np.arctan2(
        np.sin(brng)
        * np.sin(delta)
        * np.cos(lat1),
        np.cos(delta)
        - np.sin(lat1)
        * np.sin(lat2),
    )

    lon2 = (
        lon2 + np.pi
    ) % (2.0 * np.pi) - np.pi

    return (
        np.degrees(lat2),
        np.degrees(lon2),
    )


def initial_bearing(
    lat1,
    lon1,
    lat2,
    lon2,
):

    phi1 = np.radians(lat1)
    phi2 = np.radians(lat2)

    dlon = np.radians(
        lon2 - lon1
    )

    y = (
        np.sin(dlon)
        * np.cos(phi2)
    )

    x = (
        np.cos(phi1)
        * np.sin(phi2)
        - np.sin(phi1)
        * np.cos(phi2)
        * np.cos(dlon)
    )

    return (
        np.degrees(
            np.arctan2(y, x)
        )
        + 360.0
    ) % 360.0


def add_cross_track_boundaries(
    df,
    radius_columns,
):

    out = (
        df.copy()
        .sort_values("valid_time")
        .reset_index(drop=True)
    )

    if len(out) < 2:
        return out

    bearings = []

    for i in range(len(out)):

        if i == 0:
            j1 = 0
            j2 = 1

        elif i == len(out) - 1:
            j1 = i - 1
            j2 = i

        else:
            j1 = i - 1
            j2 = i + 1

        bearings.append(
            initial_bearing(
                out.loc[
                    j1,
                    "consensus_latitude",
                ],
                out.loc[
                    j1,
                    "consensus_longitude",
                ],
                out.loc[
                    j2,
                    "consensus_latitude",
                ],
                out.loc[
                    j2,
                    "consensus_longitude",
                ],
            )
        )

    out["track_bearing"] = bearings

    for radius_name in radius_columns:

        left_lat = []
        left_lon = []

        right_lat = []
        right_lon = []

        for _, row in out.iterrows():

            radius = row[radius_name]
            b = row["track_bearing"]

            lat_l, lon_l = destination_point(
                row["consensus_latitude"],
                row["consensus_longitude"],
                (b - 90.0) % 360.0,
                radius,
            )

            lat_r, lon_r = destination_point(
                row["consensus_latitude"],
                row["consensus_longitude"],
                (b + 90.0) % 360.0,
                radius,
            )

            left_lat.append(lat_l)
            left_lon.append(lon_l)

            right_lat.append(lat_r)
            right_lon.append(lon_r)

        out[
            f"{radius_name}_left_lat"
        ] = left_lat

        out[
            f"{radius_name}_left_lon"
        ] = left_lon

        out[
            f"{radius_name}_right_lat"
        ] = right_lat

        out[
            f"{radius_name}_right_lon"
        ] = right_lon

    return out


def automatic_extent(members):

    lon = members["longitude"].to_numpy(
        dtype=float
    )

    lat = members["latitude"].to_numpy(
        dtype=float
    )

    lon_min = np.nanmin(lon)
    lon_max = np.nanmax(lon)

    lat_min = np.nanmin(lat)
    lat_max = np.nanmax(lat)

    lon_pad = max(
        2.0,
        0.12 * (lon_max - lon_min),
    )

    lat_pad = max(
        2.0,
        0.12 * (lat_max - lat_min),
    )

    return [
        lon_min - lon_pad,
        lon_max + lon_pad,
        lat_min - lat_pad,
        lat_max + lat_pad,
    ]


def plot_variability_hierarchy(
    members,
    historical,
    recent,
    latest,
    recent_cycles,
    latest_cycle,
    storm,
    tracker,
    output_file,
    extent=None,
):

    # Keep heavy optional dependencies out of numerical-only runs.
    import matplotlib.pyplot as plt

    import cartopy.crs as ccrs
    import cartopy.feature as cfeature

    from cartopy.mpl.ticker import (
        LongitudeFormatter,
        LatitudeFormatter,
    )

    proj = ccrs.PlateCarree()

    model_display_names = {
        "aifs2": "AIFS2",
        "graphcast": "GraphCast",
        "pangu3": "Pangu3",
        "pangu6": "Pangu6",
    }

    # --------------------------------------------------------
    # Compare all panels over the historical/recent common
    # validity interval.
    # --------------------------------------------------------

    h_valid = historical[
        historical["spread_valid"]
    ]

    r_valid = recent[
        recent["spread_valid"]
    ]

    common_times = sorted(
        set(h_valid["valid_time"])
        & set(r_valid["valid_time"])
    )

    if len(common_times) < 2:
        raise ValueError(
            "Not enough common valid times to generate "
            "the variability comparison plot."
        )

    start = min(common_times)
    end = max(common_times)

    historical_plot = historical[
        historical["valid_time"].between(
            start,
            end,
        )
    ].copy()

    recent_plot = recent[
        recent["valid_time"].between(
            start,
            end,
        )
    ].copy()

    members_plot = members[
        members["valid_time"].between(
            start,
            end,
        )
    ].copy()

    latest_plot = latest[
        latest["valid_time"].between(
            start,
            end,
        )
    ].copy()

    recent_members = members_plot[
        members_plot["cycle"].isin(
            recent_cycles
        )
    ].copy()

    latest_members = members_plot[
        members_plot["cycle"]
        == latest_cycle
    ].copy()

    historical_plot = (
        add_cross_track_boundaries(
            historical_plot,
            ["r67_km", "r90_km"],
        )
    )

    recent_plot = (
        add_cross_track_boundaries(
            recent_plot,
            ["r67_km", "r90_km"],
        )
    )

    if extent is None:
        map_extent = automatic_extent(
            members_plot
        )
    else:
        map_extent = list(extent)

    # --------------------------------------------------------
    # Helpers
    # --------------------------------------------------------

    def setup_map(ax):

        ax.set_extent(
            map_extent,
            crs=proj,
        )

        ax.add_feature(
            cfeature.LAND.with_scale("50m"),
            facecolor="0.92",
            edgecolor="none",
            zorder=3,
        )

        ax.add_feature(
            cfeature.COASTLINE.with_scale("50m"),
            linewidth=0.8,
            edgecolor="0.15",
            zorder=8,
        )

        ax.add_feature(
            cfeature.BORDERS.with_scale("50m"),
            linewidth=0.45,
            edgecolor="0.30",
            linestyle=":",
            zorder=8,
        )

        lon0, lon1, lat0, lat1 = map_extent

        xticks = np.arange(
            np.ceil(lon0 / 5.0) * 5.0,
            lon1 + 0.01,
            5.0,
        )

        yticks = np.arange(
            np.ceil(lat0 / 2.0) * 2.0,
            lat1 + 0.01,
            2.0,
        )

        ax.set_xticks(
            xticks,
            crs=proj,
        )

        ax.set_yticks(
            yticks,
            crs=proj,
        )

        ax.xaxis.set_major_formatter(
            LongitudeFormatter(
                degree_symbol="°",
                number_format=".0f",
            )
        )

        ax.yaxis.set_major_formatter(
            LatitudeFormatter(
                degree_symbol="°",
                number_format=".0f",
            )
        )

        ax.tick_params(
            labelsize=8,
        )

        # Standard grid avoids the Cartopy/Shapely Gridliner
        # failure encountered in the Polo prototype.
        ax.grid(
            True,
            linewidth=0.45,
            alpha=0.30,
            linestyle="--",
        )


    def fill_envelope(
        ax,
        df,
        radius_name,
        alpha,
        label,
        zorder,
    ):

        lon = np.concatenate(
            [
                df[
                    f"{radius_name}_left_lon"
                ].values,

                df[
                    f"{radius_name}_right_lon"
                ].values[::-1],
            ]
        )

        lat = np.concatenate(
            [
                df[
                    f"{radius_name}_left_lat"
                ].values,

                df[
                    f"{radius_name}_right_lat"
                ].values[::-1],
            ]
        )

        ax.fill(
            lon,
            lat,
            alpha=alpha,
            label=label,
            transform=proj,
            zorder=zorder,
        )


    # --------------------------------------------------------
    # Figure
    # --------------------------------------------------------

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(21, 8),
        subplot_kw={
            "projection": proj,
        },
    )

    # ========================================================
    # A — HISTORICAL
    # ========================================================

    ax = axes[0]

    fill_envelope(
        ax,
        historical_plot,
        "r90_km",
        0.16,
        "90% spread",
        1,
    )

    fill_envelope(
        ax,
        historical_plot,
        "r67_km",
        0.27,
        "67% spread",
        2,
    )

    for (_, _), g in members_plot.groupby(
        ["cycle", "model"]
    ):

        g = g.sort_values(
            "valid_time"
        )

        ax.plot(
            g["longitude"],
            g["latitude"],
            linewidth=0.7,
            alpha=0.16,
            transform=proj,
            zorder=4,
        )

    ax.plot(
        historical_plot[
            "consensus_longitude"
        ],
        historical_plot[
            "consensus_latitude"
        ],
        linewidth=2.2,
        marker="o",
        markersize=3,
        label="Consensus",
        transform=proj,
        zorder=9,
    )

    setup_map(ax)

    ax.set_title(
        "A. Historical variability\n"
        f"All cycles | up to "
        f"{int(historical_plot['n_members'].max())} members",
        fontsize=12,
    )

    ax.legend(
        loc="upper right",
        fontsize=8,
    )

    # ========================================================
    # B — RECENT
    # ========================================================

    ax = axes[1]

    fill_envelope(
        ax,
        recent_plot,
        "r90_km",
        0.16,
        "90% spread",
        1,
    )

    fill_envelope(
        ax,
        recent_plot,
        "r67_km",
        0.27,
        "67% spread",
        2,
    )

    for (_, _), g in recent_members.groupby(
        ["cycle", "model"]
    ):

        g = g.sort_values(
            "valid_time"
        )

        ax.plot(
            g["longitude"],
            g["latitude"],
            linewidth=0.8,
            alpha=0.22,
            transform=proj,
            zorder=4,
        )

    ax.plot(
        recent_plot[
            "consensus_longitude"
        ],
        recent_plot[
            "consensus_latitude"
        ],
        linewidth=2.2,
        marker="o",
        markersize=3,
        label="Consensus",
        transform=proj,
        zorder=9,
    )

    setup_map(ax)

    ax.set_title(
        "B. Recent variability\n"
        f"Latest {len(recent_cycles)} cycles | up to "
        f"{int(recent_plot['n_members'].max())} members",
        fontsize=12,
    )

    ax.legend(
        loc="upper right",
        fontsize=8,
    )

    # ========================================================
    # C — LATEST CYCLE
    #
    # No artificial r67/r90 or rmax cone here. With only a few
    # models, explicit tracks are more informative.
    # ========================================================

    ax = axes[2]

    for model, g in latest_members.groupby(
        "model"
    ):

        g = g.sort_values(
            "valid_time"
        )

        ax.plot(
            g["longitude"],
            g["latitude"],
            linewidth=1.7,
            marker="o",
            markersize=2.8,
            label=model_display_names.get(
                str(model).lower(),
                str(model),
            ),
            transform=proj,
            zorder=5,
        )

    if not latest_plot.empty:

        ax.plot(
            latest_plot[
                "consensus_longitude"
            ],
            latest_plot[
                "consensus_latitude"
            ],
            linewidth=2.5,
            linestyle="--",
            label="Consensus",
            transform=proj,
            zorder=9,
        )

    setup_map(ax)

    ax.set_title(
        "C. Current multimodel spread\n"
        f"Latest cycle {latest_cycle} | up to "
        f"{int(latest_plot['n_members'].max())} models",
        fontsize=12,
    )

    ax.legend(
        loc="upper right",
        fontsize=8,
    )

    # --------------------------------------------------------
    # Figure text
    # --------------------------------------------------------

    fig.suptitle(
        f"AIWeather — {storm} Track Variability Hierarchy\n"
        f"Tracker: {tracker}",
        fontsize=16,
        y=0.98,
    )

    fig.text(
        0.5,
        0.025,
        "Common valid-time window: "
        f"{start:%d %b %H UTC} – "
        f"{end:%d %b %H UTC %Y}  |  "
        "Experimental forecast-spread diagnostics — "
        "not calibrated TC probability cones",
        ha="center",
        fontsize=10,
    )

    plt.tight_layout(
        rect=[
            0,
            0.055,
            1,
            0.93,
        ]
    )

    fig.savefig(
        output_file,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(fig)

    return (
        output_file,
        start,
        end,
        map_extent,
    )


# ============================================================
# MAIN
# ============================================================

def main():

    args = parse_args()

    validate_args(args)

    args.output.mkdir(
        parents=True,
        exist_ok=True,
    )

    members = pd.read_csv(
        args.members,
        parse_dates=["valid_time"],
    )

    if "init_time" in members.columns:
        members["init_time"] = pd.to_datetime(
            members["init_time"]
        )

    validate_members(members)

    # --------------------------------------------------------
    # Normalize longitude convention
    # --------------------------------------------------------
    #
    # Track products may use either 0..360 or -180..180
    # longitude.  Internally use -180..180 consistently so
    # automatic map extents and Cartopy plotting work for
    # eastern- and western-hemisphere storms alike.
    #
    members["longitude"] = normalize_longitude(
        members["longitude"].to_numpy(
            dtype=float
        )
    )

    # --------------------------------------------------------
    # Normalize cycle type
    # --------------------------------------------------------

    members["cycle"] = pd.to_numeric(
        members["cycle"],
        errors="raise",
    ).astype(int)

    cycles = sorted(
        int(cycle)
        for cycle in members["cycle"].unique()
    )

    if len(cycles) < args.recent_cycles:
        raise ValueError(
            f"Requested {args.recent_cycles} recent cycles, "
            f"but only {len(cycles)} cycles are available."
        )

    recent_cycles = cycles[
        -args.recent_cycles:
    ]

    if args.latest_cycle is None:
        latest_cycle = cycles[-1]
    else:
        latest_cycle = args.latest_cycle

        if latest_cycle not in cycles:
            raise ValueError(
                f"Requested latest cycle {latest_cycle} "
                f"is not present. Available cycles: {cycles}"
            )

    # --------------------------------------------------------
    # Subsets
    # --------------------------------------------------------

    historical_members = members.copy()

    recent_members = members[
        members["cycle"].isin(
            recent_cycles
        )
    ].copy()

    latest_members = members[
        members["cycle"] == latest_cycle
    ].copy()

    # --------------------------------------------------------
    # Statistics
    # --------------------------------------------------------

    historical = compute_variability(
        historical_members
    )

    recent = compute_variability(
        recent_members
    )

    latest = compute_variability(
        latest_members
    )

    # Historical validity:
    # retain the same concept used by the Polo prototype.
    historical["spread_valid"] = (
        (historical["n_cycles"] >= args.recent_cycles)
        & (
            historical["n_models"]
            >= args.minimum_models
        )
    )

    # Recent validity requires all requested recent cycles.
    recent["spread_valid"] = (
        (
            recent["n_cycles"]
            == len(recent_cycles)
        )
        & (
            recent["n_models"]
            >= args.minimum_models
        )
    )

    historical[
        "full_available_cycle_overlap"
    ] = (
        historical["n_cycles"]
        == len(cycles)
    )

    expected_recent_members = (
        len(recent_cycles)
        * members["model"].nunique()
    )

    recent[
        "full_recent_member_overlap"
    ] = (
        recent["n_members"]
        == expected_recent_members
    )

    # Latest-cycle r67/r90 are retained numerically in the CSV,
    # but they should NOT be presented as probabilistic envelopes
    # when the latest cycle contains only a few models.
    latest["spread_valid"] = (
        latest["n_models"] >= 2
    )

    # --------------------------------------------------------
    # Common historical/recent comparison
    # --------------------------------------------------------

    h = historical[
        [
            "valid_time",
            "n_members",
            "n_cycles",
            "n_models",
            "r67_km",
            "r90_km",
            "max_distance_km",
            "spread_valid",
        ]
    ].rename(
        columns={
            "n_members":
                "historical_n",

            "n_cycles":
                "historical_cycles",

            "n_models":
                "historical_models",

            "r67_km":
                "historical_r67_km",

            "r90_km":
                "historical_r90_km",

            "max_distance_km":
                "historical_max_km",

            "spread_valid":
                "historical_valid",
        }
    )

    r = recent[
        [
            "valid_time",
            "n_members",
            "n_cycles",
            "n_models",
            "r67_km",
            "r90_km",
            "max_distance_km",
            "spread_valid",
        ]
    ].rename(
        columns={
            "n_members":
                "recent_n",

            "n_cycles":
                "recent_cycles",

            "n_models":
                "recent_models",

            "r67_km":
                "recent_r67_km",

            "r90_km":
                "recent_r90_km",

            "max_distance_km":
                "recent_max_km",

            "spread_valid":
                "recent_valid",
        }
    )

    comparison = pd.merge(
        h,
        r,
        on="valid_time",
        how="inner",
    )

    # Avoid division by zero at zero-spread initial positions.
    comparison[
        "r67_spread_contraction_fraction"
    ] = np.where(
        comparison["historical_r67_km"] > 0.0,

        1.0
        - (
            comparison["recent_r67_km"]
            / comparison["historical_r67_km"]
        ),

        np.nan,
    )

    comparison[
        "r90_spread_contraction_fraction"
    ] = np.where(
        comparison["historical_r90_km"] > 0.0,

        1.0
        - (
            comparison["recent_r90_km"]
            / comparison["historical_r90_km"]
        ),

        np.nan,
    )

    comparison[
        "r67_spread_contraction_pct"
    ] = (
        100.0
        * comparison[
            "r67_spread_contraction_fraction"
        ]
    )

    comparison[
        "r90_spread_contraction_pct"
    ] = (
        100.0
        * comparison[
            "r90_spread_contraction_fraction"
        ]
    )

    comparison["comparison_valid"] = (
        comparison["historical_valid"]
        & comparison["recent_valid"]
    )

    # --------------------------------------------------------
    # Filenames
    # --------------------------------------------------------

    storm_slug = (
        args.storm
        .strip()
        .lower()
        .replace(" ", "_")
    )

    tracker_slug = (
        args.tracker
        .strip()
        .lower()
        .replace(" ", "_")
    )

    prefix = (
        f"{storm_slug}_{tracker_slug}"
    )

    historical_file = (
        args.output
        / f"{prefix}_variability_historical.csv"
    )

    recent_file = (
        args.output
        / (
            f"{prefix}_variability_"
            f"recent{len(recent_cycles)}.csv"
        )
    )

    latest_file = (
        args.output
        / f"{prefix}_variability_latest.csv"
    )

    comparison_file = (
        args.output
        / f"{prefix}_variability_comparison.csv"
    )

    # --------------------------------------------------------
    # Write
    # --------------------------------------------------------

    historical.to_csv(
        historical_file,
        index=False,
    )

    recent.to_csv(
        recent_file,
        index=False,
    )

    latest.to_csv(
        latest_file,
        index=False,
    )

    comparison.to_csv(
        comparison_file,
        index=False,
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    valid_compare = comparison[
        comparison["comparison_valid"]
    ].copy()

    print()
    print("=" * 78)
    print(
        f" AIWEATHER TC TRACK VARIABILITY — "
        f"{args.storm.upper()}"
    )
    print("=" * 78)

    print()
    print("Members file:")
    print(args.members)

    print()
    print("Tracker:")
    print(args.tracker)

    print()
    print("Available cycles:")
    print(cycles)

    print()
    print("Recent cycles:")
    print(recent_cycles)

    print()
    print("Latest cycle:")
    print(latest_cycle)

    print()
    print("Models:")
    print(
        sorted(
            members["model"].unique()
        )
    )

    print()
    print("-" * 78)
    print("MEMBER HIERARCHY")
    print("-" * 78)

    print(
        "Historical max members :",
        int(
            historical[
                "n_members"
            ].max()
        ),
    )

    print(
        "Recent max members     :",
        int(
            recent[
                "n_members"
            ].max()
        ),
    )

    print(
        "Latest max members     :",
        int(
            latest[
                "n_members"
            ].max()
        ),
    )

    if len(valid_compare):

        print()
        print("-" * 78)
        print("COMMON VALID COMPARISON")
        print("-" * 78)

        print(
            valid_compare[
                "valid_time"
            ].min(),
            "->",
            valid_compare[
                "valid_time"
            ].max(),
        )

        print()

        print(
            "Mean historical r67 : "
            f"{valid_compare['historical_r67_km'].mean():.1f} km"
        )

        print(
            "Mean recent r67     : "
            f"{valid_compare['recent_r67_km'].mean():.1f} km"
        )

        print(
            "Mean r67 contraction: "
            f"{valid_compare['r67_spread_contraction_pct'].mean():+.1f}%"
        )

        print()

        print(
            "Mean historical r90 : "
            f"{valid_compare['historical_r90_km'].mean():.1f} km"
        )

        print(
            "Mean recent r90     : "
            f"{valid_compare['recent_r90_km'].mean():.1f} km"
        )

        print(
            "Mean r90 contraction: "
            f"{valid_compare['r90_spread_contraction_pct'].mean():+.1f}%"
        )

    print()
    print("-" * 78)
    print("OUTPUT FILES")
    print("-" * 78)

    print(historical_file)
    print(recent_file)
    print(latest_file)
    print(comparison_file)

    print()
    print(
        "NOTE: spread statistics describe forecast disagreement; "
        "they are not calibrated TC probabilities."
    )

    if args.plots:

        plot_file = (
            args.output
            / f"{prefix}_variability_3panel.png"
        )

        (
            plot_file,
            plot_start,
            plot_end,
            map_extent,
        ) = plot_variability_hierarchy(
            members=members,
            historical=historical,
            recent=recent,
            latest=latest,
            recent_cycles=recent_cycles,
            latest_cycle=latest_cycle,
            storm=args.storm,
            tracker=args.tracker,
            output_file=plot_file,
            extent=args.extent,
        )

        print()
        print("-" * 78)
        print("PLOT")
        print("-" * 78)

        print(plot_file)

        print(
            "Common plot window:",
            plot_start,
            "->",
            plot_end,
        )

        print(
            "Map extent:",
            [round(x, 2) for x in map_extent],
        )

    print()
    print("DONE")


if __name__ == "__main__":
    main()
