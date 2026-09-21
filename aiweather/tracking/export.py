"""
Tropical cyclone track export utilities.

Persist operational and pseudo-operational AIWeather tropical
cyclone tracks together with forecast and tracking provenance.
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

from aiweather.forecast import ForecastMetadata
from aiweather.version import __version__

from .genesis import GenesisResult
from .records import TrackRecord, records_to_dataframe


def export_operational_track(
    records: list[TrackRecord],
    output_dir: str | Path,
    *,
    filename: str,
    forecast_path: str | Path,
    forecast_metadata: ForecastMetadata,
    storm_id: str,
    storm_name: str,
    seed_latitude: float,
    seed_longitude: float,
    seed_valid_time: datetime,
    seed_source: str,
    seed_source_issuance_time: datetime | None,
    experiment_type: str,
    lat_min: float,
    lat_max: float,
    lon_min: float,
    lon_max: float,
    search_radius_km: float,
    wind_radius_km: float,
    maximum_translation_speed_mps: float | None,
    datasource: str | None = None,
    datasource_source: str | None = None,
    git_commit: str | None = None,
) -> tuple[Path, Path]:
    """
    Export one tropical cyclone track and its provenance.

    Returns
    -------
    tuple[pathlib.Path, pathlib.Path]
        Paths to the track CSV and provenance JSON.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    track_path = output_dir / filename

    records_to_dataframe(records).to_csv(
        track_path,
        index=False,
    )

    initialization_time = (
        forecast_metadata.initialization_time
    )
    creation_time = (
        forecast_metadata.creation_time
    )

    manifest: dict[str, Any] = {
        "schema_version": 1,
        "aiweather_version": __version__,
        "created_at": datetime.now(
            timezone.utc
        ).isoformat(),
        "storm": {
            "id": storm_id,
            "name": storm_name,
        },
        "forecast": {
            "path": str(forecast_path),
            "forecast_id": (
                forecast_metadata.forecast_id
            ),
            "model_name": (
                forecast_metadata.model_name
            ),
            "model_version": (
                forecast_metadata.model_version
            ),
            "backend": (
                forecast_metadata.backend
            ),
            "initialization_time": (
                initialization_time.isoformat()
                if initialization_time is not None
                else None
            ),
            "creation_time": (
                creation_time.isoformat()
                if creation_time is not None
                else None
            ),
            "datasource": datasource,
            "datasource_source": datasource_source,
        },
        "seed": {
            "latitude": float(seed_latitude),
            "longitude": float(seed_longitude),
            "valid_time": seed_valid_time.isoformat(),
            "source": seed_source,
            "source_issuance_time": (
                seed_source_issuance_time.isoformat()
                if seed_source_issuance_time
                is not None
                else None
            ),
        },
        "tracking": {
            "experiment_type": experiment_type,
            "lat_min": float(lat_min),
            "lat_max": float(lat_max),
            "lon_min": float(lon_min),
            "lon_max": float(lon_max),
            "search_radius_km": float(
                search_radius_km
            ),
            "wind_radius_km": float(
                wind_radius_km
            ),
            "maximum_translation_speed_mps": (
                float(maximum_translation_speed_mps)
                if maximum_translation_speed_mps
                is not None
                else None
            ),


            "number_of_points": len(records),
            "last_lead_time_hours": (
                records[-1].lead_time_hours
                if records
                else None
            ),
            "last_valid_time": (
                str(records[-1].valid_time)
                if records
                else None
            ),
            "last_latitude": (
                float(records[-1].latitude)
                if records
                else None
            ),
            "last_longitude": (
                float(records[-1].longitude)
                if records
                else None
            ),

        },
        "software": {
            "git_commit": git_commit,
        },
    }

    provenance_path = track_path.with_suffix(
        ".provenance.json"
    )

    with provenance_path.open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            manifest,
            handle,
            indent=2,
            sort_keys=True,
        )
        handle.write("\n")

    return track_path, provenance_path


def export_native_genesis_case(
    genesis: GenesisResult,
    records: list[TrackRecord],
    output_dir: str | Path,
    *,
    filename: str,
    forecast_path: str | Path,
    forecast_metadata: ForecastMetadata,
    case_id: str,
    case_name: str | None,
    experiment_type: str,
    lat_min: float,
    lat_max: float,
    lon_min: float,
    lon_max: float,
    candidate_lead_min_hours: int,
    candidate_lead_max_hours: int,
    candidate_interval_hours: int,
    max_candidates: int,
    minimum_separation_km: float,
    maximum_displacement_km: float,
    minimum_wind: float,
    maximum_pressure: float,
    minimum_consecutive_points: int,
    wind_radius_km: float,
    search_radius_km: float,
    datasource: str | None = None,
    datasource_source: str | None = None,
    git_commit: str | None = None,
) -> tuple[Path, Path]:
    """
    Export an already-computed native genesis case and track.

    This function performs serialization only. It does not run
    genesis detection or tropical-cyclone tracking.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    track_path = output_dir / filename

    records_to_dataframe(records).to_csv(
        track_path,
        index=False,
    )

    initialization_time = (
        forecast_metadata.initialization_time
    )
    creation_time = (
        forecast_metadata.creation_time
    )

    manifest: dict[str, Any] = {
        "schema_version": 1,
        "aiweather_version": __version__,
        "created_at": datetime.now(
            timezone.utc
        ).isoformat(),
        "experiment_type": experiment_type,
        "case": {
            "id": case_id,
            "name": case_name,
        },
        "forecast": {
            "path": str(forecast_path),
            "forecast_id": (
                forecast_metadata.forecast_id
            ),
            "model_name": (
                forecast_metadata.model_name
            ),
            "model_version": (
                forecast_metadata.model_version
            ),
            "backend": (
                forecast_metadata.backend
            ),
            "initialization_time": (
                initialization_time.isoformat()
                if initialization_time is not None
                else None
            ),
            "creation_time": (
                creation_time.isoformat()
                if creation_time is not None
                else None
            ),
            "datasource": datasource,
            "datasource_source": datasource_source,
        },
        "genesis_detection": {
            "lat_min": float(lat_min),
            "lat_max": float(lat_max),
            "lon_min": float(lon_min),
            "lon_max": float(lon_max),
            "candidate_lead_min_hours": int(
                candidate_lead_min_hours
            ),
            "candidate_lead_max_hours": int(
                candidate_lead_max_hours
            ),
            "candidate_interval_hours": int(
                candidate_interval_hours
            ),
            "max_candidates": int(max_candidates),
            "minimum_separation_km": float(
                minimum_separation_km
            ),
            "maximum_displacement_km": float(
                maximum_displacement_km
            ),
            "minimum_wind": float(minimum_wind),
            "maximum_pressure": float(
                maximum_pressure
            ),
            "minimum_consecutive_points": int(
                minimum_consecutive_points
            ),
            "wind_radius_km": float(
                wind_radius_km
            ),
            "result": {
                "track_index": int(
                    genesis.track_index
                ),
                "genesis_lead_time_hours": int(
                    genesis.genesis_lead_time_hours
                ),
                "latitude": float(
                    genesis.latitude
                ),
                "longitude": float(
                    genesis.longitude
                ),
                "pressure": float(
                    genesis.pressure
                ),
                "max_wind": float(
                    genesis.max_wind
                ),
                "qualifying_points": int(
                    genesis.qualifying_points
                ),
            },
        },
        "post_genesis_tracking": {
            "search_radius_km": float(
                search_radius_km
            ),
            "wind_radius_km": float(
                wind_radius_km
            ),
            "number_of_points": len(records),
            "last_lead_time_hours": (
                records[-1].lead_time_hours
                if records
                else None
            ),
            "last_valid_time": (
                str(records[-1].valid_time)
                if records
                else None
            ),
            "last_latitude": (
                float(records[-1].latitude)
                if records
                else None
            ),
            "last_longitude": (
                float(records[-1].longitude)
                if records
                else None
            ),
        },
        "software": {
            "git_commit": git_commit,
        },
    }

    provenance_path = track_path.with_suffix(
        ".provenance.json"
    )

    with provenance_path.open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            manifest,
            handle,
            indent=2,
            sort_keys=True,
        )
        handle.write("\n")

    return track_path, provenance_path
