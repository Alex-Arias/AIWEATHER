"""
Verification export utilities.

Persist tropical cyclone tracking and verification results so that
plotting and downstream analysis can run without repeating GPU-based
tracker execution.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from aiweather.tracking.records import (
    TrackRecord,
    records_to_dataframe,
)
from aiweather.tracking.workflow import (
    TrackingWorkflowResult,
)

from .comparison import (
    TrackVerification,
    common_overlap_verifications,
)
from .workflow import (
    VerificationWorkflowResult,
)

import json
from datetime import datetime, timezone
from typing import Any

from aiweather.version import __version__

def _write_records(
    records: list[TrackRecord],
    path: Path,
) -> None:
    """
    Write TrackRecord objects to CSV.
    """
    dataframe = records_to_dataframe(
        records
    )

    dataframe.to_csv(
        path,
        index=False,
    )


def _write_verification(
    verification: TrackVerification,
    path: Path,
) -> None:
    """
    Write one verification table to CSV.
    """
    verification.table.to_csv(
        path,
        index=False,
    )

def export_verification_manifest(
    output_dir: str | Path,
    *,
    forecast_path: str | Path,
    forecast_metadata,
    sid: str,
    lat_min: float,
    lat_max: float,
    lon_min: float,
    lon_max: float,
    device: str,
    minimum_overlap: int,
    maximum_mean_error_km: float | None = None,
    ibtracs_path: str | Path,
    ibtracs_basin: str,
    generate_plots: bool,
    field_lead_times: list[
        int | float
    ] | None = None,
) -> Path:
    """
    Export provenance metadata for one tropical cyclone
    verification case.

    Returns
    -------
    pathlib.Path
        Path to ``run_manifest.json``.
    """
    output_dir = Path(
        output_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
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
        "created_at": (
            datetime.now(
                timezone.utc
            ).isoformat()
        ),
        "forecast": {
            "path": str(
                forecast_path
            ),
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
                if initialization_time
                is not None
                else None
            ),
            "creation_time": (
                creation_time.isoformat()
                if creation_time
                is not None
                else None
            ),
        },
        "verification": {
            "sid": sid,
            "lat_min": float(
                lat_min
            ),
            "lat_max": float(
                lat_max
            ),
            "lon_min": float(
                lon_min
            ),
            "lon_max": float(
                lon_max
            ),
            "device": device,
            "minimum_overlap": int(
                minimum_overlap
            ),
            "maximum_mean_error_km": (
                float(
                    maximum_mean_error_km
                )
                if maximum_mean_error_km
                is not None
                else None
            ),
        },
        "ibtracs": {
            "basin": ibtracs_basin,
            "path": str(
                ibtracs_path
            ),
        },
        "plots": {
            "enabled": bool(
                generate_plots
            ),
            "field_lead_times_hours": (
                list(
                    field_lead_times
                )
                if field_lead_times
                is not None
                else None
            ),
        },
    }

    manifest_path = (
        output_dir
        / "run_manifest.json"
    )

    with manifest_path.open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            manifest,
            handle,
            indent=2,
            sort_keys=True,
        )

        handle.write(
            "\n"
        )

    return manifest_path

def export_verification_case(
    output_dir: str | Path,
    *,
    tracking_result: TrackingWorkflowResult,
    verification_result: VerificationWorkflowResult,
) -> Path:
    """
    Export one tropical cyclone verification case.

    Parameters
    ----------
    output_dir
        Destination directory.

    tracking_result
        Output from the tropical cyclone tracking workflow.

    verification_result
        Verification results against best-track observations.

    Returns
    -------
    pathlib.Path
        Export directory.
    """

    if not isinstance(
        tracking_result,
        TrackingWorkflowResult,
    ):
        raise TypeError(
            "tracking_result must be a "
            "TrackingWorkflowResult."
        )

    if not isinstance(
        verification_result,
        VerificationWorkflowResult,
    ):
        raise TypeError(
            "verification_result must be a "
            "VerificationWorkflowResult."
        )

    output_dir = Path(
        output_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ---------------------------------------------------------
    # Track records
    # ---------------------------------------------------------

    _write_records(
        verification_result.observations,
        output_dir / "ibtracs.csv",
    )

    _write_records(
        tracking_result.native_records,
        output_dir / "native.csv",
    )

    evaluation = tracking_result.evaluation

    if (
        evaluation is not None
        and evaluation.wuduan_match is not None
    ):
        _write_records(
            evaluation.wuduan_match.records,
            output_dir / "wuduan.csv",
        )

    if (
        evaluation is not None
        and evaluation.vitart_match is not None
    ):
        _write_records(
            evaluation.vitart_match.records,
            output_dir / "vitart.csv",
        )

    # ---------------------------------------------------------
    # Full verification
    # ---------------------------------------------------------

    for (
        name,
        verification,
    ) in verification_result.verifications.items():
        _write_verification(
            verification,
            output_dir
            / f"verification_{name}.csv",
        )

    # ---------------------------------------------------------
    # Common-overlap verification
    # ---------------------------------------------------------

    common = common_overlap_verifications(
        verification_result.verifications
    )

    for (
        name,
        verification,
    ) in common.items():
        _write_verification(
            verification,
            output_dir
            / f"common_{name}.csv",
        )

    # ---------------------------------------------------------
    # Summary table
    # ---------------------------------------------------------

    summary_rows = []

    for (
        name,
        verification,
    ) in verification_result.verifications.items():
        summary_rows.append(
            {
                "tracker": name,
                "coverage": "full",
                "overlap_count":
                    verification.overlap_count,
                "mean_track_error_km":
                    verification.mean_track_error_km,
                "rmse_track_error_km":
                    verification.rmse_track_error_km,
                "median_track_error_km":
                    verification.median_track_error_km,
                "maximum_track_error_km":
                    verification.maximum_track_error_km,
                "pressure_mae_pa":
                    verification
                    .mean_absolute_pressure_error_pa,
                "pressure_rmse_pa":
                    verification
                    .rmse_pressure_error_pa,
                "wind_mae_ms":
                    verification
                    .mean_absolute_wind_error_ms,
                "wind_rmse_ms":
                    verification
                    .rmse_wind_error_ms,
            }
        )

    for (
        name,
        verification,
    ) in common.items():
        summary_rows.append(
            {
                "tracker": name,
                "coverage": "common",
                "overlap_count":
                    verification.overlap_count,
                "mean_track_error_km":
                    verification.mean_track_error_km,
                "rmse_track_error_km":
                    verification.rmse_track_error_km,
                "median_track_error_km":
                    verification.median_track_error_km,
                "maximum_track_error_km":
                    verification.maximum_track_error_km,
                "pressure_mae_pa":
                    verification
                    .mean_absolute_pressure_error_pa,
                "pressure_rmse_pa":
                    verification
                    .rmse_pressure_error_pa,
                "wind_mae_ms":
                    verification
                    .mean_absolute_wind_error_ms,
                "wind_rmse_ms":
                    verification
                    .rmse_wind_error_ms,
            }
        )

    pd.DataFrame(
        summary_rows
    ).to_csv(
        output_dir
        / "verification_summary.csv",
        index=False,
    )

    return output_dir