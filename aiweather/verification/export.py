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