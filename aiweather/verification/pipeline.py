"""
End-to-end tropical cyclone verification pipeline.

This module orchestrates forecast loading, tropical cyclone tracking,
IBTrACS best-track verification, persistence of verification products,
and optional generation of standard verification figures without
reimplementing lower-level tracking, verification, or plotting logic.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from aiweather.forecast import open_forecast
from aiweather.plotting import (
    plot_pressure_evolution,
    plot_track_error,
    plot_track_map,
    plot_wind_evolution,
)
from aiweather.tracking.workflow import (
    TrackingWorkflowResult,
    evaluate_forecast_trackers,
)

from .best_track import (
    best_track_to_records,
)
from .comparison import (
    common_overlap_verifications,
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
    plot_paths: dict[str, Path] = field(
        default_factory=dict
    )


def _verification_tables(
    verification: VerificationWorkflowResult,
) -> dict:
    """
    Return available verification tables keyed by tracker name.
    """
    return {
        name.capitalize(): result.table
        for (
            name,
            result,
        ) in verification.verifications.items()
    }


def _common_verification_tables(
    verification: VerificationWorkflowResult,
) -> dict:
    """
    Return common-overlap verification tables.
    """
    common = common_overlap_verifications(
        verification.verifications
    )

    return {
        name.capitalize(): result.table
        for (
            name,
            result,
        ) in common.items()
    }


def _forecast_track_records(
    tracking: TrackingWorkflowResult,
) -> dict:
    """
    Return available forecast tracks for map plotting.
    """
    forecasts = {
        "Native": tracking.native_records,
    }

    evaluation = tracking.evaluation

    if (
        evaluation is not None
        and evaluation.wuduan_match is not None
    ):
        forecasts[
            "WuDuan"
        ] = evaluation.wuduan_match.records

    if (
        evaluation is not None
        and evaluation.vitart_match is not None
    ):
        forecasts[
            "Vitart"
        ] = evaluation.vitart_match.records

    return forecasts


def _generate_verification_plots(
    *,
    tracking: TrackingWorkflowResult,
    verification: VerificationWorkflowResult,
    plot_output_dir: Path,
    sid: str,
) -> dict[str, Path]:
    """
    Generate standard non-field tropical cyclone verification plots.
    """
    import matplotlib.pyplot as plt

    plot_output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    plot_paths: dict[str, Path] = {}

    forecasts = _forecast_track_records(
        tracking
    )

    verification_tables = (
        _verification_tables(
            verification
        )
    )

    common_tables = (
        _common_verification_tables(
            verification
        )
    )

    # ---------------------------------------------------------
    # Track map
    # ---------------------------------------------------------

    figure, _ = plot_track_map(
        verification.observations,
        forecasts=forecasts,
        title=(
            f"Tropical cyclone tracks "
            f"vs IBTrACS {sid}"
        ),
        annotate_lead_time=True,
        lead_time_interval_hours=24,
    )

    figure.tight_layout()

    path = (
        plot_output_dir
        / "track_map.png"
    )

    figure.savefig(
        path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(
        figure
    )

    plot_paths[
        "track_map"
    ] = path

    # ---------------------------------------------------------
    # Full track error
    # ---------------------------------------------------------

    figure, _ = plot_track_error(
        verification_tables,
        title=(
            f"Track error vs IBTrACS "
            f"{sid}"
        ),
    )

    figure.tight_layout()

    path = (
        plot_output_dir
        / "track_error.png"
    )

    figure.savefig(
        path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(
        figure
    )

    plot_paths[
        "track_error"
    ] = path

    # ---------------------------------------------------------
    # Common-overlap track error
    # ---------------------------------------------------------

    figure, _ = plot_track_error(
        common_tables,
        title=(
            f"Common-overlap track error "
            f"vs IBTrACS {sid}"
        ),
    )

    figure.tight_layout()

    path = (
        plot_output_dir
        / "track_error_common.png"
    )

    figure.savefig(
        path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(
        figure
    )

    plot_paths[
        "track_error_common"
    ] = path

    # ---------------------------------------------------------
    # Pressure evolution
    # ---------------------------------------------------------

    figure, _ = plot_pressure_evolution(
        verification_tables,
        title=(
            f"Minimum central pressure "
            f"vs IBTrACS {sid}"
        ),
    )

    figure.tight_layout()

    path = (
        plot_output_dir
        / "pressure_evolution.png"
    )

    figure.savefig(
        path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(
        figure
    )

    plot_paths[
        "pressure_evolution"
    ] = path

    # ---------------------------------------------------------
    # Wind evolution
    # ---------------------------------------------------------

    figure, _ = plot_wind_evolution(
        verification_tables,
        title=(
            f"Maximum wind vs IBTrACS "
            f"{sid}"
        ),
    )

    figure.tight_layout()

    path = (
        plot_output_dir
        / "wind_evolution.png"
    )

    figure.savefig(
        path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(
        figure
    )

    plot_paths[
        "wind_evolution"
    ] = path

    return plot_paths


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
    generate_plots: bool = False,
    plot_output_dir: str | Path | None = None,
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

    generate_plots
        Generate standard verification figures when True.

    plot_output_dir
        Optional directory for generated figures.

        If omitted while ``generate_plots=True``:

        * ``output_dir / "plots"`` is used when output_dir is
          available;
        * otherwise ``results/plots/<sid>`` is used.

    Returns
    -------
    TCVerificationPipelineResult
        Tracking result, verification result, optional
        export directory, and generated plot paths.
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

    if not isinstance(
        generate_plots,
        bool,
    ):
        raise TypeError(
            "generate_plots must be a boolean."
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

    plot_paths: dict[str, Path] = {}

    if generate_plots:

        if plot_output_dir is None:
            if output_dir is not None:
                resolved_plot_output_dir = (
                    Path(output_dir)
                    / "plots"
                )
            else:
                resolved_plot_output_dir = (
                    Path("results")
                    / "plots"
                    / sid
                )
        else:
            resolved_plot_output_dir = Path(
                plot_output_dir
            )

        plot_paths = _generate_verification_plots(
            tracking=tracking,
            verification=verification,
            plot_output_dir=(
                resolved_plot_output_dir
            ),
            sid=sid,
        )

    return TCVerificationPipelineResult(
        tracking=tracking,
        verification=verification,
        output_dir=exported,
        plot_paths=plot_paths,
    )