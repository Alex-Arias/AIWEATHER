"""
End-to-end tropical cyclone verification pipeline.

This module orchestrates forecast loading, tropical cyclone tracking,
IBTrACS best-track verification, and persistence of verification
products without reimplementing lower-level tracking or verification
logic.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from aiweather.forecast import open_forecast
from aiweather.tracking.workflow import (
    TrackingWorkflowResult,
    evaluate_forecast_trackers,
)

from .best_track import (
    best_track_to_records,
)
from .export import (
    export_verification_case,
)
from .ibtracs import (
    read_ibtracs_csv,
)
from .workflow import (
    VerificationWorkflowResult,
    verify_tracking_workflow,
)


@dataclass(slots=True)
class TCVerificationPipelineResult:
    """
    Result of an end-to-end tropical cyclone verification run.
    """

    tracking: TrackingWorkflowResult
    verification: VerificationWorkflowResult
    output_dir: Path | None = None


def run_tc_verification_pipeline(
    forecast_path: str | Path,
    *,
    ibtracs_path: str | Path,
    sid: str,
    lat_min: float,
    lat_max: float,
    lon_min: float,
    lon_max: float,
    device: str = "cpu",
    minimum_overlap: int = 1,
    output_dir: str | Path | None = None,
) -> TCVerificationPipelineResult:
    """
    Run tropical cyclone tracking and best-track verification.

    Parameters
    ----------
    forecast_path
        Path to the AIWeather forecast store.

    ibtracs_path
        Path to an IBTrACS CSV file.

    sid
        IBTrACS storm identifier.

    lat_min, lat_max, lon_min, lon_max
        Tracking-domain bounds.

    device
        Device passed to the external tracker workflow.

    minimum_overlap
        Minimum overlap required when matching tracker paths.

    output_dir
        Optional directory where verification CSV products
        are exported.

    Returns
    -------
    TCVerificationPipelineResult
        Tracking result, verification result, and optional
        export directory.
    """
    if minimum_overlap < 1:
        raise ValueError(
            "minimum_overlap must be at least 1."
        )

    if not isinstance(
        sid,
        str,
    ):
        raise TypeError(
            "sid must be a string."
        )

    if not sid.strip():
        raise ValueError(
            "sid cannot be empty."
        )

    forecast = open_forecast(
        forecast_path
    )

    tracking = evaluate_forecast_trackers(
        forecast,
        lat_min=lat_min,
        lat_max=lat_max,
        lon_min=lon_min,
        lon_max=lon_max,
        device=device,
        minimum_overlap=minimum_overlap,
    )

    best_track_points = read_ibtracs_csv(
        ibtracs_path,
        sid=sid,
    )

    best_track_records = best_track_to_records(
        best_track_points,
        initialization_time=(
            forecast.metadata.initialization_time
        ),
    )

    verification = verify_tracking_workflow(
        tracking,
        best_track_records,
        observation_name=f"ibtracs_{sid}",
    )

    exported = None

    if output_dir is not None:
        exported = export_verification_case(
            output_dir,
            tracking_result=tracking,
            verification_result=verification,
        )

    return TCVerificationPipelineResult(
        tracking=tracking,
        verification=verification,
        output_dir=exported,
    )