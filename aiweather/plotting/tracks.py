"""
Tropical cyclone track plotting utilities.

Provides plotting helpers for observed and forecast tropical cyclone
tracks using the standardized AIWeather TrackRecord representation.
"""

from __future__ import annotations

from collections.abc import Mapping

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from aiweather.tracking.records import TrackRecord


def normalize_longitude(
    longitude,
):
    """
    Normalize longitude to the [-180, 180) convention.

    Parameters
    ----------
    longitude
        Scalar or array-like longitude in degrees.

    Returns
    -------
    float or numpy.ndarray
        Longitude expressed in the [-180, 180) range.
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
        return float(normalized)

    return normalized


def _validate_records(
    records: list[TrackRecord],
    *,
    name: str,
) -> None:
    """
    Validate a TrackRecord collection.
    """
    if not isinstance(
        records,
        list,
    ):
        raise TypeError(
            f"{name} must be a list."
        )

    for record in records:
        if not isinstance(
            record,
            TrackRecord,
        ):
            raise TypeError(
                f"{name} must contain "
                "TrackRecord objects."
            )


def _track_coordinates(
    records: list[TrackRecord],
) -> tuple[
    np.ndarray,
    np.ndarray,
]:
    """
    Extract normalized longitude and latitude arrays.
    """
    longitude = normalize_longitude(
        [
            record.longitude
            for record in records
        ]
    )

    latitude = np.asarray(
        [
            record.latitude
            for record in records
        ],
        dtype=float,
    )

    return (
        longitude,
        latitude,
    )


def plot_track_map(
    observations: list[TrackRecord],
    *,
    forecasts: Mapping[
        str,
        list[TrackRecord],
    ] | None = None,
    ax: Axes | None = None,
    title: str | None = None,
    annotate_lead_time: bool = False,
    lead_time_interval_hours: int = 24,
    longitude_margin: float = 3.0,
    latitude_margin: float = 3.0,
) -> tuple[
    Figure,
    Axes,
]:
    """
    Plot observed and forecast tropical cyclone tracks.

    Parameters
    ----------
    observations
        Observed best-track records.

    forecasts
        Mapping from forecast/tracker name to TrackRecord list.

        Example::

            {
                "Native": native_records,
                "WuDuan": wuduan_records,
                "Vitart": vitart_records,
            }

    ax
        Existing Matplotlib/Cartopy axes. If omitted, a new
        PlateCarree geographic axes is created.

    title
        Optional plot title.

    annotate_lead_time
        Annotate selected forecast points with forecast lead time.

    lead_time_interval_hours
        Interval used for lead-time labels.

    longitude_margin
        Longitude padding added around all plotted tracks.

    latitude_margin
        Latitude padding added around all plotted tracks.

    Returns
    -------
    Figure, Axes
        Matplotlib figure and Cartopy geographic axes.
    """
    _validate_records(
        observations,
        name="observations",
    )

    if forecasts is None:
        forecasts = {}

    if not isinstance(
        forecasts,
        Mapping,
    ):
        raise TypeError(
            "forecasts must be a mapping."
        )

    for name, records in forecasts.items():
        if not isinstance(
            name,
            str,
        ):
            raise TypeError(
                "forecast names must be strings."
            )

        _validate_records(
            records,
            name=f"forecasts[{name!r}]",
        )

    if lead_time_interval_hours <= 0:
        raise ValueError(
            "lead_time_interval_hours must be "
            "greater than zero."
        )

    if longitude_margin < 0.0:
        raise ValueError(
            "longitude_margin cannot be negative."
        )

    if latitude_margin < 0.0:
        raise ValueError(
            "latitude_margin cannot be negative."
        )

    geographic_crs = ccrs.PlateCarree()

    # ---------------------------------------------------------
    # Figure / axes
    # ---------------------------------------------------------

    if ax is None:
        figure = plt.figure(
            figsize=(9, 7),
        )

        ax = figure.add_subplot(
            1,
            1,
            1,
            projection=geographic_crs,
        )
    else:
        figure = ax.figure

    all_longitudes = []
    all_latitudes = []

    # ---------------------------------------------------------
    # Observations
    # ---------------------------------------------------------

    if observations:
        longitude, latitude = (
            _track_coordinates(
                observations
            )
        )

        all_longitudes.extend(
            longitude.tolist()
        )

        all_latitudes.extend(
            latitude.tolist()
        )

        ax.plot(
            longitude,
            latitude,
            marker="o",
            linewidth=2.0,
            markersize=4.0,
            label="IBTrACS",
            transform=geographic_crs,
        )

    # ---------------------------------------------------------
    # Forecast trackers
    # ---------------------------------------------------------

    for name, records in forecasts.items():
        if not records:
            continue

        longitude, latitude = (
            _track_coordinates(
                records
            )
        )

        all_longitudes.extend(
            longitude.tolist()
        )

        all_latitudes.extend(
            latitude.tolist()
        )

        ax.plot(
            longitude,
            latitude,
            marker="o",
            linewidth=1.6,
            markersize=3.5,
            label=name,
            transform=geographic_crs,
        )

        if annotate_lead_time:
            for (
                record,
                lon,
                lat,
            ) in zip(
                records,
                longitude,
                latitude,
            ):
                lead = int(
                    record.lead_time_hours
                )

                if (
                    lead
                    % lead_time_interval_hours
                    != 0
                ):
                    continue

                ax.annotate(
                    f"+{lead}h",
                    xy=(lon, lat),
                    xytext=(4, 4),
                    textcoords="offset points",
                    fontsize=8,
                )

    # ---------------------------------------------------------
    # Geographic extent
    # ---------------------------------------------------------

    if all_longitudes and all_latitudes:
        lon_min = (
            min(all_longitudes)
            - longitude_margin
        )

        lon_max = (
            max(all_longitudes)
            + longitude_margin
        )

        lat_min = (
            min(all_latitudes)
            - latitude_margin
        )

        lat_max = (
            max(all_latitudes)
            + latitude_margin
        )

        ax.set_extent(
            [
                lon_min,
                lon_max,
                lat_min,
                lat_max,
            ],
            crs=geographic_crs,
        )

    # ---------------------------------------------------------
    # Geographic context
    # ---------------------------------------------------------

    ax.add_feature(
        cfeature.LAND,
        alpha=0.25,
    )

    ax.add_feature(
        cfeature.COASTLINE,
        linewidth=0.8,
    )

    ax.add_feature(
        cfeature.BORDERS,
        linewidth=0.5,
    )

    ax.gridlines(
        draw_labels=False,
        linewidth=0.5,
        alpha=0.3,
    )


    # ---------------------------------------------------------
    # Presentation
    # ---------------------------------------------------------

    if title is not None:
        ax.set_title(
            title
        )

    if observations or any(
        forecasts.values()
    ):
        ax.legend()

    return (
        figure,
        ax,
    )
