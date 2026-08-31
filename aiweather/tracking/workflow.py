"""
High-level tropical cyclone tracking workflow.

This module orchestrates native AIWeather tropical cyclone tracking
and optional Earth2Studio tracker evaluation for an existing forecast.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from aiweather.diagnostics import detect_pressure_minima
from aiweather.forecast import Forecast

from .association import associate_candidates
from .evaluation import TrackerEvaluation
from .genesis import (
    GenesisResult,
    detect_genesis,
    select_first_genesis,
)
from .records import (
    TrackRecord,
    build_track_records,
)
from .tropical_cyclone import (
    track_from_genesis,
    track_pressure_minimum,
)


@dataclass(slots=True)
class TrackingWorkflowResult:
    """
    Result of the complete tropical cyclone tracking workflow.
    """

    genesis: GenesisResult | None
    native_records: list[TrackRecord]
    evaluation: TrackerEvaluation | None = None

def build_existing_tc_track(
    forecast: Forecast,
    *,
    initial_latitude: float,
    initial_longitude: float,
    lat_min: float,
    lat_max: float,
    lon_min: float,
    lon_max: float,
    search_radius_km: float = 500.0,
    wind_radius_km: float = 300.0,
    maximum_translation_speed_mps: float | None = 20.0,
) -> list[TrackRecord]:
    """
    Build a native track for an existing tropical cyclone.

    Tracking begins at forecast lead zero from a known operational
    storm center. Unlike ``build_native_tc_track``, this workflow
    does not perform genesis detection.

    Tracking terminates when the optional maximum translation
    speed is exceeded.
    """

    region = forecast.select_region(
        lat_min=lat_min,
        lat_max=lat_max,
        lon_min=lon_min,
        lon_max=lon_max,
    )

    track = track_pressure_minimum(
        region["msl"],
        initial_latitude=initial_latitude,
        initial_longitude=initial_longitude,
        start_index=0,
        u_wind=region["u10m"],
        v_wind=region["v10m"],
        search_radius_km=search_radius_km,
        wind_radius_km=wind_radius_km,
        maximum_translation_speed_mps=(
            maximum_translation_speed_mps
        ),
    )

    initialization_time = (
        forecast.metadata.initialization_time
    )

    if initialization_time is None:
        raise ValueError(
            "Forecast initialization_time metadata "
            "is required for track records."
        )

    return build_track_records(
        track,
        initialization_time=initialization_time,
    )


def build_native_tc_track(
    forecast: Forecast,
    *,
    lat_min: float,
    lat_max: float,
    lon_min: float,
    lon_max: float,
    candidate_lead_min_hours: int = 24,
    candidate_lead_max_hours: int = 174,
    candidate_interval_hours: int = 6,
    max_candidates: int = 5,
    minimum_separation_km: float = 500.0,
    maximum_displacement_km: float = 500.0,
    minimum_wind: float = 17.0,
    maximum_pressure: float = 100500.0,
    minimum_consecutive_points: int = 3,
    wind_radius_km: float = 300.0,
    search_radius_km: float = 500.0,
) -> tuple[
    GenesisResult,
    list[TrackRecord],
]:
    """
    Build the native AIWeather tropical cyclone track.

    Returns
    -------
    GenesisResult
        Selected genesis result.

    list[TrackRecord]
        Native AIWeather tropical cyclone track records.
    """

    region = forecast.select_region(
        lat_min=lat_min,
        lat_max=lat_max,
        lon_min=lon_min,
        lon_max=lon_max,
    )

    lead_values = (
        region["lead_time"]
        .values
        .astype("timedelta64[h]")
        .astype(int)
    )

    candidates_by_lead = []

    for lead_hours in range(
        candidate_lead_min_hours,
        candidate_lead_max_hours + 1,
        candidate_interval_hours,
    ):
        matches = np.flatnonzero(
            lead_values == lead_hours
        )

        if len(matches) == 0:
            continue

        pressure = region["msl"].isel(
            time=0,
            lead_time=int(matches[0]),
        )

        minima = detect_pressure_minima(
            pressure,
            max_candidates=max_candidates,
            minimum_separation_km=(
                minimum_separation_km
            ),
        )

        candidates_by_lead.append(
            (
                lead_hours,
                minima,
            )
        )

    candidate_tracks = associate_candidates(
        candidates_by_lead,
        maximum_displacement_km=(
            maximum_displacement_km
        ),
    )

    genesis_results = detect_genesis(
        candidate_tracks,
        u_wind=region["u10m"],
        v_wind=region["v10m"],
        minimum_wind=minimum_wind,
        maximum_pressure=maximum_pressure,
        minimum_consecutive_points=(
            minimum_consecutive_points
        ),
        wind_radius_km=wind_radius_km,
    )

    genesis = select_first_genesis(
        genesis_results
    )

    if genesis is None:
        raise RuntimeError(
            "No tropical cyclone genesis was detected "
            "in the selected forecast region."
        )

    native_track = track_from_genesis(
        region["msl"],
        genesis=genesis,
        u_wind=region["u10m"],
        v_wind=region["v10m"],
        search_radius_km=search_radius_km,
        wind_radius_km=wind_radius_km,
    )

    initialization_time = (
        forecast.metadata.initialization_time
    )

    if initialization_time is None:
        raise ValueError(
            "Forecast initialization_time metadata "
            "is required for track records."
        )

    native_records = build_track_records(
        native_track,
        initialization_time=initialization_time,
    )

    return (
        genesis,
        native_records,
    )

def evaluate_forecast_trackers(
    forecast: Forecast,
    *,
    lat_min: float,
    lat_max: float,
    lon_min: float,
    lon_max: float,
    dataset=None,
    device: str = "cuda",
    minimum_overlap: int = 3,
    maximum_mean_error_km: float | None = None,
    run_wuduan: bool = True,
    run_vitart: bool = True,
    reference_records: list[TrackRecord] | None = None,
    **native_kwargs,
) -> TrackingWorkflowResult:
    """
    Run the complete AIWeather tropical cyclone tracking workflow.

    The native AIWeather track is built first. Optional Earth2Studio
    WuDuan and Vitart trackers are then run and automatically matched
    against the native reference track.

    Parameters
    ----------
    forecast
        AIWeather Forecast object.

    lat_min, lat_max, lon_min, lon_max
        Region used for native tropical cyclone tracking.

    dataset
        xarray Dataset used by Earth2Studio trackers. If omitted,
        the dataset stored in ``forecast`` is used.

    device
        Device passed to Earth2Studio trackers.

    minimum_overlap
        Minimum number of common lead times required when matching
        Earth2Studio tracks to the native reference.

    run_wuduan, run_vitart
        Enable or disable individual Earth2Studio trackers.

    **native_kwargs
        Additional arguments forwarded to ``build_native_tc_track``.

    Returns
    -------
    TrackingWorkflowResult
        Native genesis, native track records, and tracker evaluation.
    """
    from .earth2studio import (
        run_vitart_tracker,
        run_wuduan_tracker,
    )
    from .evaluation import compare_tracker_ensemble

    try:
        genesis, native_records = build_native_tc_track(
            forecast,
            lat_min=lat_min,
            lat_max=lat_max,
            lon_min=lon_min,
            lon_max=lon_max,
            **native_kwargs,
        )
    except RuntimeError as exc:
        no_genesis = (
            "No tropical cyclone genesis was detected"
            in str(exc)
        )

        if (
            not no_genesis
            or reference_records is None
        ):
            raise

        genesis = None
        native_records = []

    initialization_time = (
        forecast.metadata.initialization_time
    )

    if initialization_time is None:
        raise ValueError(
            "Forecast initialization_time metadata "
            "is required for tracker evaluation."
        )

    if dataset is None:
        dataset = forecast.dataset

    wuduan_tracks = None

    if run_wuduan:
        wuduan_tracks = run_wuduan_tracker(
            dataset,
            device=device,
        )

    vitart_tracks = None

    if run_vitart:
        vitart_tracks = run_vitart_tracker(
            dataset,
            device=device,
        )

    matching_reference = (
        native_records
        if reference_records is None
        else reference_records
    )

    evaluation = compare_tracker_ensemble(
        matching_reference,
        initialization_time=initialization_time,
        wuduan_tracks=wuduan_tracks,
        vitart_tracks=vitart_tracks,
        minimum_overlap=minimum_overlap,
        maximum_mean_error_km=maximum_mean_error_km,
    )

    return TrackingWorkflowResult(
        genesis=genesis,
        native_records=native_records,
        evaluation=evaluation,
    )