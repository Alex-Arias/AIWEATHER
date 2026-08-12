"""
Earth2Studio tropical-cyclone tracker integration.

This module provides optional wrappers around Earth2Studio diagnostic
trackers while keeping the core AIWeather tracking package independent
of Earth2Studio's optional cyclone dependencies.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import xarray as xr

from .records import TrackRecord


@dataclass(frozen=True, slots=True)
class Earth2StudioTrack:
    """
    Standardized representation of one Earth2Studio TC path.

    Attributes
    ----------
    path_id : int
        Path identifier assigned by Earth2Studio.

    lead_time_hours : np.ndarray
        Forecast lead times associated with valid path points.

    latitude : np.ndarray
        Storm-center latitude.

    longitude : np.ndarray
        Storm-center longitude in degrees east.

    pressure : np.ndarray
        Minimum sea-level pressure.

    max_wind : np.ndarray
        Maximum 10-m wind associated with the center.
    """

    path_id: int
    lead_time_hours: np.ndarray
    latitude: np.ndarray
    longitude: np.ndarray
    pressure: np.ndarray
    max_wind: np.ndarray

    def __len__(self) -> int:
        return int(self.lead_time_hours.size)


def _load_tracker_class(name: str):
    """
    Import an Earth2Studio TC tracker lazily.
    """
    try:
        from earth2studio.models.dx import (
            TCTrackerVitart,
            TCTrackerWuDuan,
        )
    except Exception as exc:
        raise ImportError(
            "Earth2Studio cyclone tracking dependencies are not "
            "available. Install the Earth2Studio 'cyclone' extra."
        ) from exc

    mapping = {
        "wuduan": TCTrackerWuDuan,
        "vitart": TCTrackerVitart,
    }

    try:
        return mapping[name]
    except KeyError as exc:
        raise ValueError(
            f"Unknown Earth2Studio tracker: {name!r}."
        ) from exc


def _validate_dataset(
    dataset: xr.Dataset,
) -> None:
    """
    Validate coordinates needed by Earth2Studio trackers.
    """
    if not isinstance(dataset, xr.Dataset):
        raise TypeError(
            "dataset must be an xarray.Dataset."
        )

    required_coords = {
        "time",
        "lead_time",
        "lat",
        "lon",
    }

    missing = required_coords.difference(
        dataset.coords
    )

    if missing:
        raise ValueError(
            "Dataset is missing required coordinates: "
            + ", ".join(sorted(missing))
        )

    if dataset.sizes["time"] != 1:
        raise ValueError(
            "Earth2Studio integration currently requires exactly "
            "one initialization time."
        )


def _check_tracker_grid(
    dataset: xr.Dataset,
    expected_coords: OrderedDict,
) -> None:
    """
    Ensure the dataset grid matches the Earth2Studio tracker grid.
    """
    expected_lat = np.asarray(
        expected_coords["lat"]
    )

    expected_lon = np.asarray(
        expected_coords["lon"]
    )

    actual_lat = np.asarray(
        dataset["lat"].values
    )

    actual_lon = np.asarray(
        dataset["lon"].values
    )

    if actual_lat.shape != expected_lat.shape:
        raise ValueError(
            "Dataset latitude grid does not match "
            "Earth2Studio tracker requirements."
        )

    if actual_lon.shape != expected_lon.shape:
        raise ValueError(
            "Dataset longitude grid does not match "
            "Earth2Studio tracker requirements."
        )

    if not np.allclose(
        actual_lat,
        expected_lat,
    ):
        raise ValueError(
            "Dataset latitude coordinates do not match "
            "Earth2Studio tracker requirements."
        )

    if not np.allclose(
        actual_lon,
        expected_lon,
    ):
        raise ValueError(
            "Dataset longitude coordinates do not match "
            "Earth2Studio tracker requirements."
        )


def _build_input_tensor(
    dataset: xr.Dataset,
    *,
    lead_index: int,
    variables: list[str],
    device: torch.device,
) -> torch.Tensor:
    """
    Construct one Earth2Studio input snapshot.

    Output shape is:

        [batch, variable, lat, lon]
    """
    arrays = []

    for name in variables:
        if name not in dataset.data_vars:
            raise ValueError(
                f"Dataset does not contain required "
                f"Earth2Studio variable {name!r}."
            )

        da = dataset[name].isel(
            time=0,
            lead_time=lead_index,
        )

        values = np.asarray(
            da.compute().values,
            dtype=np.float32,
        )

        arrays.append(
            values
        )

    stacked = np.stack(
        arrays,
        axis=0,
    )[None, ...]

    return torch.from_numpy(
        stacked
    ).to(
        device=device
    )


def _extract_paths(
    tracker: Any,
    output_coords: OrderedDict,
    lead_hours: np.ndarray,
) -> list[Earth2StudioTrack]:
    """
    Convert Earth2Studio path_buffer into standardized tracks.
    """
    paths = (
        tracker.path_buffer
        .detach()
        .cpu()
        .numpy()
    )

    variables = list(
        output_coords["variable"]
    )

    lat_index = variables.index(
        "tclat"
    )

    lon_index = variables.index(
        "tclon"
    )

    pressure_index = variables.index(
        "tcmsl"
    )

    wind_index = variables.index(
        "tcw10m"
    )

    fill_value = float(
        tracker.PATH_FILL_VALUE
    )

    results: list[
        Earth2StudioTrack
    ] = []

    for path_id in range(
        paths.shape[1]
    ):
        path = paths[
            0,
            path_id,
            :,
            :,
        ]

        valid = (
            np.isfinite(
                path[:, lat_index]
            )
            & np.isfinite(
                path[:, lon_index]
            )
            & (
                path[:, lat_index]
                != fill_value
            )
            & (
                path[:, lon_index]
                != fill_value
            )
        )

        if not np.any(valid):
            continue

        results.append(
            Earth2StudioTrack(
                path_id=path_id,
                lead_time_hours=np.asarray(
                    lead_hours[valid],
                    dtype=int,
                ),
                latitude=np.asarray(
                    path[valid, lat_index],
                    dtype=float,
                ),
                longitude=np.asarray(
                    path[valid, lon_index],
                    dtype=float,
                ),
                pressure=np.asarray(
                    path[
                        valid,
                        pressure_index,
                    ],
                    dtype=float,
                ),
                max_wind=np.asarray(
                    path[
                        valid,
                        wind_index,
                    ],
                    dtype=float,
                ),
            )
        )

    return results


def run_earth2studio_tracker(
    dataset: xr.Dataset,
    *,
    tracker_name: str,
    device: str | torch.device = "cuda",
    **tracker_kwargs,
) -> list[Earth2StudioTrack]:
    """
    Run an Earth2Studio TC tracker over all forecast lead times.

    Parameters
    ----------
    dataset : xr.Dataset
        Global AIWeather forecast dataset.

    tracker_name : {"wuduan", "vitart"}
        Earth2Studio tracker implementation.

    device : str or torch.device, default="cuda"
        Torch device used for tracker execution.

    **tracker_kwargs
        Keyword arguments forwarded to the Earth2Studio tracker.

    Returns
    -------
    list[Earth2StudioTrack]
        All paths accumulated by the tracker.
    """
    _validate_dataset(
        dataset
    )

    tracker_class = _load_tracker_class(
        tracker_name
    )

    tracker = tracker_class(
        **tracker_kwargs
    )

    input_coords = (
        tracker.input_coords()
    )

    _check_tracker_grid(
        dataset,
        input_coords,
    )

    variables = list(
        input_coords["variable"]
    )

    for variable in variables:
        if variable not in dataset.data_vars:
            raise ValueError(
                f"Dataset is missing required "
                f"variable {variable!r}."
            )

    torch_device = torch.device(
        device
    )

    if (
        torch_device.type == "cuda"
        and not torch.cuda.is_available()
    ):
        raise RuntimeError(
            "CUDA was requested but is not available."
        )

    lead_hours = (
        dataset["lead_time"]
        .values
        .astype("timedelta64[h]")
        .astype(int)
    )

    coords = OrderedDict(
        [
            (
                "batch",
                np.array([0]),
            ),
            (
                "variable",
                np.asarray(
                    variables,
                    dtype=object,
                ),
            ),
            (
                "lat",
                dataset[
                    "lat"
                ].values,
            ),
            (
                "lon",
                dataset[
                    "lon"
                ].values,
            ),
        ]
    )

    output_coords = None

    tracker.reset_path_buffer()

    for lead_index in range(
        len(lead_hours)
    ):
        tensor = _build_input_tensor(
            dataset,
            lead_index=lead_index,
            variables=variables,
            device=torch_device,
        )

        with torch.no_grad():
            _, output_coords = tracker(
                tensor,
                coords,
            )

    if output_coords is None:
        return []

    return _extract_paths(
        tracker,
        output_coords,
        lead_hours,
    )


def run_wuduan_tracker(
    dataset: xr.Dataset,
    *,
    device: str | torch.device = "cuda",
    **tracker_kwargs,
) -> list[Earth2StudioTrack]:
    """
    Run Earth2Studio TCTrackerWuDuan.
    """
    return run_earth2studio_tracker(
        dataset,
        tracker_name="wuduan",
        device=device,
        **tracker_kwargs,
    )


def run_vitart_tracker(
    dataset: xr.Dataset,
    *,
    device: str | torch.device = "cuda",
    **tracker_kwargs,
) -> list[Earth2StudioTrack]:
    """
    Run Earth2Studio TCTrackerVitart.
    """
    return run_earth2studio_tracker(
        dataset,
        tracker_name="vitart",
        device=device,
        **tracker_kwargs,
    )


def select_regional_track(
    tracks: list[Earth2StudioTrack],
    *,
    lat_min: float,
    lat_max: float,
    lon_min: float,
    lon_max: float,
    lead_min_hours: int | None = None,
    lead_max_hours: int | None = None,
) -> Earth2StudioTrack | None:
    """
    Select the track with the largest number of points in a region.

    Optional lead-time bounds may be supplied to restrict the
    selection to a forecast-time window.

    Parameters
    ----------
    tracks : list[Earth2StudioTrack]
        Earth2Studio tracks to search.

    lat_min, lat_max : float
        Latitude bounds.

    lon_min, lon_max : float
        Longitude bounds in degrees east.

    lead_min_hours : int, optional
        Ignore track points before this lead time.

    lead_max_hours : int, optional
        Ignore track points after this lead time.

    Returns
    -------
    Earth2StudioTrack or None
        Track containing the largest number of qualifying points.
    """
    if lat_min > lat_max:
        raise ValueError(
            "lat_min must be less than or equal to lat_max."
        )

    if lon_min > lon_max:
        raise ValueError(
            "lon_min must be less than or equal to lon_max."
        )

    if (
        lead_min_hours is not None
        and lead_max_hours is not None
        and lead_min_hours > lead_max_hours
    ):
        raise ValueError(
            "lead_min_hours must be less than or equal to "
            "lead_max_hours."
        )

    best = None
    best_count = 0

    for track in tracks:
        inside = (
            (track.latitude >= lat_min)
            & (track.latitude <= lat_max)
            & (track.longitude >= lon_min)
            & (track.longitude <= lon_max)
        )

        if lead_min_hours is not None:
            inside &= (
                track.lead_time_hours
                >= lead_min_hours
            )

        if lead_max_hours is not None:
            inside &= (
                track.lead_time_hours
                <= lead_max_hours
            )

        count = int(
            inside.sum()
        )

        if count > best_count:
            best = track
            best_count = count

    return best

def earth2studio_track_to_records(
    track: Earth2StudioTrack,
    *,
    initialization_time,
) -> list[TrackRecord]:
    """
    Convert an Earth2Studio path to AIWeather TrackRecord objects.

    The Earth2Studio path is first converted to AIWeather TrackPoint
    objects, then passed through the existing build_track_records()
    pipeline so that motion diagnostics are calculated consistently
    with native AIWeather tracks.
    """
    from .records import build_track_records
    from .tropical_cyclone import TrackPoint

    points = [
        TrackPoint(
            lead_time_hours=int(lead),
            latitude=float(latitude),
            longitude=float(longitude),
            pressure=float(pressure),
            pressure_units="Pa",
            max_wind=float(max_wind),
            wind_units="m/s",
        )
        for (
            lead,
            latitude,
            longitude,
            pressure,
            max_wind,
        ) in zip(
            track.lead_time_hours,
            track.latitude,
            track.longitude,
            track.pressure,
            track.max_wind,
        )
    ]

    return build_track_records(
        points,
        initialization_time=initialization_time,
    )
