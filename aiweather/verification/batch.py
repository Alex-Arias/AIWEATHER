"""
Batch tropical cyclone verification.

This module provides orchestration utilities for running the
AIWeather tropical cyclone verification pipeline across multiple
forecast cases.

The batch layer deliberately reuses the tested single-case
verification pipeline rather than duplicating tracking or
verification logic.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from aiweather.forecast import open_forecast

from .comparison import (
    common_overlap_verifications,
)
from .pipeline import (
    TCVerificationPipelineResult,
    run_tc_verification_pipeline,
)


@dataclass(slots=True)
class TCVerificationCase:
    """
    Description of one tropical cyclone verification case.

    Parameters
    ----------
    forecast_path
        Path to an AIWeather forecast store.

    sid
        IBTrACS storm identifier.

    lat_min, lat_max, lon_min, lon_max
        Tropical cyclone tracking-domain bounds.

    case_id
        Optional user-defined identifier for the case.

        This is useful when the same storm is evaluated from
        multiple forecast initialization times.
    """

    forecast_path: str | Path
    sid: str
    lat_min: float
    lat_max: float
    lon_min: float
    lon_max: float
    case_id: str | None = None


@dataclass(slots=True)
class TCVerificationBatchResult:
    """
    Result of a multi-case tropical cyclone verification run.

    Parameters
    ----------
    cases
        Verification cases that were executed.

    results
        Single-case verification pipeline results.

    summary
        Aggregate per-case verification summary.
    """

    cases: list[TCVerificationCase]
    results: list[TCVerificationPipelineResult]
    summary: pd.DataFrame
    output_dir: Path | None = None
    summary_path: Path | None = None


def _validate_case(
    case: TCVerificationCase,
) -> None:
    """
    Validate one batch verification case.
    """

    if not isinstance(
        case,
        TCVerificationCase,
    ):
        raise TypeError(
            "case must be a TCVerificationCase."
        )

    if not isinstance(
        case.sid,
        str,
    ):
        raise TypeError(
            "case.sid must be a string."
        )

    if not case.sid.strip():
        raise ValueError(
            "case.sid cannot be empty."
        )

    if (
        case.lat_min
        >= case.lat_max
    ):
        raise ValueError(
            "case.lat_min must be less than "
            "case.lat_max."
        )

    if (
        case.lon_min
        >= case.lon_max
    ):
        raise ValueError(
            "case.lon_min must be less than "
            "case.lon_max."
        )

    if (
        case.case_id is not None
        and not isinstance(
            case.case_id,
            str,
        )
    ):
        raise TypeError(
            "case.case_id must be a string "
            "or None."
        )

    if (
        isinstance(
            case.case_id,
            str,
        )
        and not case.case_id.strip()
    ):
        raise ValueError(
            "case.case_id cannot be empty."
        )


def _verification_summary_rows(
    *,
    case: TCVerificationCase,
    result: TCVerificationPipelineResult,
) -> list[dict]:
    """
    Build aggregate summary rows for one verification case.
    """

    forecast = open_forecast(
        case.forecast_path
    )

    metadata = forecast.metadata

    initialization_time = (
        metadata.initialization_time
    )

    rows: list[dict] = []

    verification = result.verification

    full = verification.verifications

    common = common_overlap_verifications(
        full
    )

    coverage_sets = (
        ("full", full),
        ("common", common),
    )

    for (
        coverage,
        verifications,
    ) in coverage_sets:

        for (
            tracker_name,
            tracker_verification,
        ) in verifications.items():

            rows.append(
                {
                    "case_id": case.case_id,
                    "model_name":
                        metadata.model_name,
                    "model_version":
                        metadata.model_version,
                    "backend":
                        metadata.backend,
                    "forecast_id":
                        metadata.forecast_id,
                    "forecast_path":
                        str(case.forecast_path),
                    "sid":
                        case.sid,
                    "initialization_time":
                        initialization_time,
                    "tracker":
                        tracker_name,
                    "coverage":
                        coverage,
                    "overlap_count":
                        tracker_verification
                        .overlap_count,
                    "mean_track_error_km":
                        tracker_verification
                        .mean_track_error_km,
                    "rmse_track_error_km":
                        tracker_verification
                        .rmse_track_error_km,
                    "median_track_error_km":
                        tracker_verification
                        .median_track_error_km,
                    "maximum_track_error_km":
                        tracker_verification
                        .maximum_track_error_km,
                    "pressure_mae_pa":
                        tracker_verification
                        .mean_absolute_pressure_error_pa,
                    "pressure_rmse_pa":
                        tracker_verification
                        .rmse_pressure_error_pa,
                    "wind_mae_ms":
                        tracker_verification
                        .mean_absolute_wind_error_ms,
                    "wind_rmse_ms":
                        tracker_verification
                        .rmse_wind_error_ms,
                }
            )

    return rows


def run_tc_verification_batch(
    cases: list[TCVerificationCase],
    *,
    device: str = "cpu",
    minimum_overlap: int = 1,
    generate_plots: bool = False,
    field_lead_times: list[
        int | float
    ] | None = None,
    ibtracs_path: str | Path | None = None,
    ibtracs_basin: str = "EP",
    ibtracs_cache_dir: str | Path = (
        "data/verification/ibtracs"
    ),
    ibtracs_max_age_hours: float = 48.0,
    ibtracs_force_update: bool = False,
    output_dir: str | Path | None = None,
) -> TCVerificationBatchResult:
    """
    Run tropical cyclone verification for multiple cases.

    Each case is processed using the standard tested
    ``run_tc_verification_pipeline`` function.

    Parameters
    ----------
    cases
        Tropical cyclone verification cases.

    device
        Tracking device, for example ``"cpu"`` or ``"cuda"``.

    minimum_overlap
        Minimum overlap required for tracker-path matching.

    generate_plots
        Generate standard single-case verification plots.

    field_lead_times
        Optional forecast lead times used for field-sequence plots.

    ibtracs_path
        Optional explicit IBTrACS CSV path.

    ibtracs_basin
        Basin used for automatic IBTrACS resolution.

    ibtracs_cache_dir
        Local IBTrACS cache directory.

    ibtracs_max_age_hours
        Maximum acceptable age of the cached IBTrACS dataset.

    ibtracs_force_update
        Force refresh of automatically managed IBTrACS data.

    output_dir
        Optional directory for batch-level products.

        When provided, AIWeather writes the aggregate
        verification table to ``batch_summary.csv``.

        When None, no batch-level files are written.

    Returns
    -------
    TCVerificationBatchResult
        Individual pipeline results and aggregate summary.
    """

    if not isinstance(
        cases,
        list,
    ):
        raise TypeError(
            "cases must be a list."
        )

    if not cases:
        raise ValueError(
            "cases cannot be empty."
        )

    for case in cases:
        _validate_case(
            case
        )

    results: list[
        TCVerificationPipelineResult
    ] = []

    summary_rows: list[dict] = []

    for case in cases:

        result = run_tc_verification_pipeline(
            case.forecast_path,
            sid=case.sid,
            lat_min=case.lat_min,
            lat_max=case.lat_max,
            lon_min=case.lon_min,
            lon_max=case.lon_max,
            device=device,
            minimum_overlap=minimum_overlap,
            generate_plots=generate_plots,
            field_lead_times=field_lead_times,
            ibtracs_path=ibtracs_path,
            ibtracs_basin=ibtracs_basin,
            ibtracs_cache_dir=(
                ibtracs_cache_dir
            ),
            ibtracs_max_age_hours=(
                ibtracs_max_age_hours
            ),
            ibtracs_force_update=(
                ibtracs_force_update
            ),
        )

        results.append(
            result
        )

        summary_rows.extend(
            _verification_summary_rows(
                case=case,
                result=result,
            )
        )

    summary = pd.DataFrame(
        summary_rows
    )

    resolved_output_dir = None
    summary_path = None

    if output_dir is not None:
        resolved_output_dir = Path(
            output_dir
        )

        resolved_output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        summary_path = (
            resolved_output_dir
            / "batch_summary.csv"
        )

        summary.to_csv(
            summary_path,
            index=False,
        )

    return TCVerificationBatchResult(
        cases=list(cases),
        results=results,
        summary=summary,
        output_dir=resolved_output_dir,
        summary_path=summary_path,
    )