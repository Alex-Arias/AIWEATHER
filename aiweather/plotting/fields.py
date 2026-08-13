"""
Meteorological field plotting utilities.

Provides tropical-cyclone field maps using 10-m wind speed,
10-m wind vectors, mean sea-level pressure, optional
storm-center overlays, and multi-panel field sequences.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize


def wind_speed(
    u_wind: Any,
    v_wind: Any,
):
    """
    Return horizontal wind speed from zonal and meridional
    wind components.
    """
    return np.hypot(
        u_wind,
        v_wind,
    )


def _select_lead_time(
    field: xr.DataArray,
    lead_time_hours: int | float,
) -> xr.DataArray:
    """
    Select one forecast lead time from a DataArray.
    """
    if not isinstance(
        field,
        xr.DataArray,
    ):
        raise TypeError(
            "field must be an xarray DataArray."
        )

    if "lead_time" not in field.dims:
        return field

    lead_coordinate = field[
        "lead_time"
    ]

    if np.issubdtype(
        lead_coordinate.dtype,
        np.timedelta64,
    ):
        target = np.timedelta64(
            int(lead_time_hours),
            "h",
        )

        return field.sel(
            lead_time=target,
        )

    values = np.asarray(
        lead_coordinate.values
    )

    if (
        values.size > 0
        and np.issubdtype(
            values.dtype,
            np.number,
        )
        and np.nanmax(values) > 40
    ):
        return field.sel(
            lead_time=lead_time_hours,
            method="nearest",
        )

    index = int(
        round(
            float(lead_time_hours)
            / 6.0
        )
    )

    if (
        index < 0
        or index >= field.sizes["lead_time"]
    ):
        raise IndexError(
            "lead_time_hours is outside the "
            "available forecast range."
        )

    return field.isel(
        lead_time=index,
    )


def _squeeze_field(
    field: xr.DataArray,
) -> xr.DataArray:
    """
    Remove singleton dimensions while preserving lat/lon.
    """
    field = field.squeeze(
        drop=True
    )

    required = {
        "lat",
        "lon",
    }

    if not required.issubset(
        field.dims
    ):
        raise ValueError(
            "field must contain lat and lon "
            "dimensions."
        )

    extra_dims = [
        dim
        for dim in field.dims
        if dim not in required
    ]

    if extra_dims:
        raise ValueError(
            "field contains unsupported "
            f"dimensions: {extra_dims}"
        )

    return field.transpose(
        "lat",
        "lon",
    )


def _normalize_longitude(
    longitude,
):
    """
    Normalize longitude to [-180, 180).
    """
    values = np.asarray(
        longitude,
        dtype=float,
    )

    normalized = (
        (values + 180.0)
        % 360.0
    ) - 180.0

    if normalized.ndim == 0:
        return float(
            normalized
        )

    return normalized


def _prepare_fields(
    dataset: xr.Dataset,
    lead_time_hours: int | float,
) -> tuple[
    xr.DataArray,
    xr.DataArray,
    xr.DataArray,
]:
    """
    Select and align U10, V10, and MSLP fields.
    """
    u_wind = _squeeze_field(
        _select_lead_time(
            dataset["u10m"],
            lead_time_hours,
        )
    )

    v_wind = _squeeze_field(
        _select_lead_time(
            dataset["v10m"],
            lead_time_hours,
        )
    )

    pressure = _squeeze_field(
        _select_lead_time(
            dataset["msl"],
            lead_time_hours,
        )
    )

    return xr.align(
        u_wind,
        v_wind,
        pressure,
        join="exact",
    )


def storm_centered_extent(
    latitude: float,
    longitude: float,
    *,
    latitude_margin: float = 8.0,
    longitude_margin: float = 12.0,
) -> tuple[
    float,
    float,
    float,
    float,
]:
    """
    Build a plotting extent around a tropical cyclone center.
    """
    latitude = float(
        latitude
    )

    longitude = float(
        _normalize_longitude(
            longitude
        )
    )

    if not np.isfinite(
        latitude
    ):
        raise ValueError(
            "latitude must be finite."
        )

    if not np.isfinite(
        longitude
    ):
        raise ValueError(
            "longitude must be finite."
        )

    if latitude_margin <= 0.0:
        raise ValueError(
            "latitude_margin must be positive."
        )

    if longitude_margin <= 0.0:
        raise ValueError(
            "longitude_margin must be positive."
        )

    return (
        longitude - longitude_margin,
        longitude + longitude_margin,
        latitude - latitude_margin,
        latitude + latitude_margin,
    )


def plot_tc_field(
    dataset: xr.Dataset,
    *,
    lead_time_hours: int | float,
    ax: Axes | None = None,
    lat_min: float | None = None,
    lat_max: float | None = None,
    lon_min: float | None = None,
    lon_max: float | None = None,
    quiver_stride: int = 8,
    pressure_interval_hpa: float = 4.0,
    centers: Mapping[
        str,
        tuple[float, float],
    ] | None = None,
    title: str | None = None,
    add_colorbar: bool = True,
    wind_speed_limits: tuple[
        float,
        float,
    ] | None = None,
) -> tuple[
    Figure,
    Axes,
]:
    """
    Plot a tropical-cyclone meteorological field.

    The map contains 10-m wind-speed shading, wind vectors,
    MSLP contours, and optional tropical-cyclone centers.
    """
    if not isinstance(
        dataset,
        xr.Dataset,
    ):
        raise TypeError(
            "dataset must be an xarray Dataset."
        )

    required = {
        "u10m",
        "v10m",
        "msl",
    }

    missing = required.difference(
        dataset.data_vars
    )

    if missing:
        raise ValueError(
            "dataset is missing required "
            f"variables: {sorted(missing)}"
        )

    if quiver_stride < 1:
        raise ValueError(
            "quiver_stride must be at least 1."
        )

    if pressure_interval_hpa <= 0:
        raise ValueError(
            "pressure_interval_hpa must be positive."
        )

    if not isinstance(
        add_colorbar,
        bool,
    ):
        raise TypeError(
            "add_colorbar must be a boolean."
        )

    if wind_speed_limits is not None:
        if (
            not isinstance(
                wind_speed_limits,
                tuple,
            )
            or len(
                wind_speed_limits
            ) != 2
        ):
            raise TypeError(
                "wind_speed_limits must be a "
                "(minimum, maximum) tuple."
            )

        wind_min = float(
            wind_speed_limits[0]
        )

        wind_max = float(
            wind_speed_limits[1]
        )

        if (
            not np.isfinite(wind_min)
            or not np.isfinite(wind_max)
        ):
            raise ValueError(
                "wind_speed_limits must be finite."
            )

        if wind_min < 0.0:
            raise ValueError(
                "wind-speed minimum cannot be negative."
            )

        if wind_max <= wind_min:
            raise ValueError(
                "wind-speed maximum must be "
                "greater than minimum."
            )
    else:
        wind_min = None
        wind_max = None

    if centers is None:
        centers = {}

    if not isinstance(
        centers,
        Mapping,
    ):
        raise TypeError(
            "centers must be a mapping."
        )

    for name, center in centers.items():
        if not isinstance(
            name,
            str,
        ):
            raise TypeError(
                "center names must be strings."
            )

        if (
            not isinstance(
                center,
                tuple,
            )
            or len(center) != 2
        ):
            raise TypeError(
                "center values must be "
                "(latitude, longitude) tuples."
            )

        latitude_value = float(
            center[0]
        )

        longitude_value = float(
            center[1]
        )

        if not np.isfinite(
            latitude_value
        ):
            raise ValueError(
                "center latitude must be finite."
            )

        if not np.isfinite(
            longitude_value
        ):
            raise ValueError(
                "center longitude must be finite."
            )

    (
        u_wind,
        v_wind,
        pressure,
    ) = _prepare_fields(
        dataset,
        lead_time_hours,
    )

    speed = wind_speed(
        u_wind,
        v_wind,
    )

    longitude = np.asarray(
        u_wind["lon"].values
    )

    latitude = np.asarray(
        u_wind["lat"].values
    )

    if ax is None:
        try:
            import cartopy.crs as ccrs

            figure = plt.figure(
                figsize=(10, 7),
            )

            ax = figure.add_subplot(
                1,
                1,
                1,
                projection=ccrs.PlateCarree(),
            )

            transform = (
                ccrs.PlateCarree()
            )

            use_cartopy = True

        except ImportError:
            figure, ax = plt.subplots(
                figsize=(10, 7),
            )

            transform = None
            use_cartopy = False

    else:
        figure = ax.figure

        try:
            import cartopy.crs as ccrs

            use_cartopy = hasattr(
                ax,
                "projection",
            )

            transform = (
                ccrs.PlateCarree()
                if use_cartopy
                else None
            )

        except ImportError:
            use_cartopy = False
            transform = None

    plot_kwargs = {}

    if transform is not None:
        plot_kwargs[
            "transform"
        ] = transform

    shading = ax.pcolormesh(
        longitude,
        latitude,
        speed,
        shading="auto",
        vmin=wind_min,
        vmax=wind_max,
        **plot_kwargs,
    )

    if add_colorbar:
        colorbar = figure.colorbar(
            shading,
            ax=ax,
            pad=0.03,
        )

        colorbar.set_label(
            "10-m wind speed [m/s]"
        )

    pressure_hpa = (
        pressure / 100.0
    )

    pressure_values = np.asarray(
        pressure_hpa
    )

    p_min = (
        np.floor(
            np.nanmin(
                pressure_values
            )
            / pressure_interval_hpa
        )
        * pressure_interval_hpa
    )

    p_max = (
        np.ceil(
            np.nanmax(
                pressure_values
            )
            / pressure_interval_hpa
        )
        * pressure_interval_hpa
    )

    levels = np.arange(
        p_min,
        p_max
        + pressure_interval_hpa,
        pressure_interval_hpa,
    )

    contours = ax.contour(
        longitude,
        latitude,
        pressure_hpa,
        levels=levels,
        linewidths=0.8,
        **plot_kwargs,
    )

    ax.clabel(
        contours,
        inline=True,
        fontsize=8,
        fmt="%.0f",
    )

    stride = int(
        quiver_stride
    )

    ax.quiver(
        longitude[
            ::stride
        ],
        latitude[
            ::stride
        ],
        np.asarray(
            u_wind
        )[
            ::stride,
            ::stride
        ],
        np.asarray(
            v_wind
        )[
            ::stride,
            ::stride
        ],
        **plot_kwargs,
    )

    for (
        name,
        center,
    ) in centers.items():

        center_latitude = float(
            center[0]
        )

        center_longitude = (
            _normalize_longitude(
                center[1]
            )
        )

        scatter_kwargs = {
            "s": 70,
            "marker": "o",
            "edgecolors": "black",
            "linewidths": 0.8,
            "label": name,
            "zorder": 10,
        }

        if transform is not None:
            scatter_kwargs[
                "transform"
            ] = transform

        ax.scatter(
            center_longitude,
            center_latitude,
            **scatter_kwargs,
        )

    if use_cartopy:
        import cartopy.feature as cfeature

        ax.coastlines(
            resolution="50m",
            linewidth=0.8,
        )

        ax.add_feature(
            cfeature.BORDERS,
            linewidth=0.5,
        )

        if (
            lat_min is not None
            and lat_max is not None
            and lon_min is not None
            and lon_max is not None
        ):
            ax.set_extent(
                [
                    _normalize_longitude(
                        lon_min
                    ),
                    _normalize_longitude(
                        lon_max
                    ),
                    lat_min,
                    lat_max,
                ],
                crs=transform,
            )

    else:
        ax.set_xlabel(
            "Longitude"
        )

        ax.set_ylabel(
            "Latitude"
        )

        if (
            lon_min is not None
            and lon_max is not None
        ):
            ax.set_xlim(
                lon_min,
                lon_max,
            )

        if (
            lat_min is not None
            and lat_max is not None
        ):
            ax.set_ylim(
                lat_min,
                lat_max,
            )

    if title is None:
        title = (
            "10-m wind and mean sea-level pressure "
            f"(+{lead_time_hours:g} h)"
        )

    ax.set_title(
        title
    )

    if centers:
        ax.legend(
            loc="upper right",
        )

    return (
        figure,
        ax,
    )


def plot_tc_field_sequence(
    dataset: xr.Dataset,
    *,
    lead_times: list[int | float],
    centers_by_lead: Mapping[
        int | float,
        Mapping[
            str,
            tuple[float, float],
        ],
    ] | None = None,
    reference_centers: Mapping[
        int | float,
        tuple[float, float],
    ] | None = None,
    latitude_margin: float = 8.0,
    longitude_margin: float = 12.0,
    quiver_stride: int = 16,
    pressure_interval_hpa: float = 4.0,
    ncols: int = 2,
    figsize: tuple[
        float,
        float,
    ] = (
        14.0,
        10.0,
    ),
) -> tuple[
    Figure,
    np.ndarray,
]:
    """
    Plot a sequence of tropical-cyclone field maps.

    All panels use one common 10-m wind-speed scale and a
    single shared colorbar.
    """
    if not isinstance(
        dataset,
        xr.Dataset,
    ):
        raise TypeError(
            "dataset must be an xarray Dataset."
        )

    if not isinstance(
        lead_times,
        list,
    ):
        raise TypeError(
            "lead_times must be a list."
        )

    if not lead_times:
        raise ValueError(
            "lead_times cannot be empty."
        )

    if ncols < 1:
        raise ValueError(
            "ncols must be at least 1."
        )

    if centers_by_lead is None:
        centers_by_lead = {}

    if reference_centers is None:
        reference_centers = {}

    if not isinstance(
        centers_by_lead,
        Mapping,
    ):
        raise TypeError(
            "centers_by_lead must be a mapping."
        )

    if not isinstance(
        reference_centers,
        Mapping,
    ):
        raise TypeError(
            "reference_centers must be a mapping."
        )

    try:
        import cartopy.crs as ccrs

        projection = ccrs.PlateCarree()

    except ImportError as exc:
        raise ImportError(
            "Cartopy is required for "
            "plot_tc_field_sequence."
        ) from exc

    # ---------------------------------------------------------
    # Shared wind-speed scale
    # ---------------------------------------------------------

    sequence_maximum = 0.0

    for lead in lead_times:
        (
            u_wind,
            v_wind,
            _,
        ) = _prepare_fields(
            dataset,
            lead,
        )

        speed = wind_speed(
            u_wind,
            v_wind,
        )

        maximum = float(
            np.nanmax(
                np.asarray(
                    speed
                )
            )
        )

        sequence_maximum = max(
            sequence_maximum,
            maximum,
        )

    if (
        not np.isfinite(
            sequence_maximum
        )
        or sequence_maximum <= 0.0
    ):
        sequence_maximum = 1.0

    shared_limits = (
        0.0,
        sequence_maximum,
    )

    count = len(
        lead_times
    )

    nrows = int(
        np.ceil(
            count / ncols
        )
    )

    figure, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=figsize,
        subplot_kw={
            "projection": projection,
        },
        squeeze=False,
    )

    flat_axes = axes.ravel()

    for index, lead in enumerate(
        lead_times
    ):
        ax = flat_axes[
            index
        ]

        centers = dict(
            centers_by_lead.get(
                lead,
                {},
            )
        )

        reference_center = (
            reference_centers.get(
                lead
            )
        )

        if (
            reference_center is None
            and "IBTrACS" in centers
        ):
            reference_center = (
                centers["IBTrACS"]
            )

        if (
            reference_center is None
            and "Native" in centers
        ):
            reference_center = (
                centers["Native"]
            )

        extent = None

        if reference_center is not None:
            extent = (
                storm_centered_extent(
                    reference_center[0],
                    reference_center[1],
                    latitude_margin=(
                        latitude_margin
                    ),
                    longitude_margin=(
                        longitude_margin
                    ),
                )
            )

        if extent is None:
            lon_min = None
            lon_max = None
            lat_min = None
            lat_max = None
        else:
            (
                lon_min,
                lon_max,
                lat_min,
                lat_max,
            ) = extent

        plot_tc_field(
            dataset,
            lead_time_hours=lead,
            ax=ax,
            lat_min=lat_min,
            lat_max=lat_max,
            lon_min=lon_min,
            lon_max=lon_max,
            quiver_stride=quiver_stride,
            pressure_interval_hpa=(
                pressure_interval_hpa
            ),
            centers=centers,
            title=(
                f"+{lead:g} h"
            ),
            add_colorbar=False,
            wind_speed_limits=(
                shared_limits
            ),
        )

        ax.text(
            0.02,
            0.98,
            chr(
                ord("a")
                + index
            ),
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=12,
            fontweight="bold",
            zorder=20,
        )

    for index in range(
        count,
        len(flat_axes),
    ):
        flat_axes[
            index
        ].set_visible(
            False
        )

    # ---------------------------------------------------------
    # Shared colorbar
    # ---------------------------------------------------------

    scalar_mappable = ScalarMappable(
        norm=Normalize(
            vmin=shared_limits[0],
            vmax=shared_limits[1],
        ),
    )

    scalar_mappable.set_array(
        []
    )

    visible_axes = [
        flat_axes[index]
        for index in range(
            count
        )
    ]

    colorbar = figure.colorbar(
        scalar_mappable,
        ax=visible_axes,
        orientation="horizontal",
        fraction=0.04,
        pad=0.06,
        aspect=35,
    )

    colorbar.set_label(
        "10-m wind speed [m/s]"
    )

    figure.subplots_adjust(
       left=0.06,
        right=0.96,
        top=0.94,
        bottom=0.12,
        wspace=0.08,
        hspace=0.12,
    )

    return (
        figure,
        axes,
    )