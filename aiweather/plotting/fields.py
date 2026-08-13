"""
Meteorological field plotting utilities.

Provides tropical-cyclone field maps using 10-m wind speed,
10-m wind vectors, mean sea-level pressure, and optional
storm-center overlays.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib.axes import Axes
from matplotlib.figure import Figure


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
) -> tuple[
    Figure,
    Axes,
]:
    """
    Plot a tropical-cyclone meteorological field.

    The map contains:

    * 10-m wind-speed shading;
    * 10-m horizontal wind vectors;
    * mean sea-level pressure contours;
    * optional tropical-cyclone center overlays.

    Parameters
    ----------
    dataset
        Forecast dataset containing ``u10m``, ``v10m``,
        and ``msl``.

    lead_time_hours
        Forecast lead time in hours.

    ax
        Optional Matplotlib or Cartopy axes.

    lat_min, lat_max, lon_min, lon_max
        Optional map bounds.

    quiver_stride
        Grid-point stride used for wind vectors.

    pressure_interval_hpa
        MSLP contour interval in hPa.

    centers
        Optional mapping from center name to
        ``(latitude, longitude)``.

        Example::

            {
                "Native": (13.0, 254.0),
                "WuDuan": (12.9, 253.9),
                "IBTrACS": (12.6, -107.6),
            }

    title
        Optional figure title.

    Returns
    -------
    figure, axes
        Matplotlib figure and axes.
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

    u_wind, v_wind, pressure = xr.align(
        u_wind,
        v_wind,
        pressure,
        join="exact",
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
        **plot_kwargs,
    )

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

    # ---------------------------------------------------------
    # Tropical cyclone centers
    # ---------------------------------------------------------

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
        ax.legend()

    return (
        figure,
        ax,
    )