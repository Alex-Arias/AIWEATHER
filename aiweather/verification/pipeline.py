"""
End-to-end tropical cyclone verification pipeline.

This module orchestrates forecast loading, tropical cyclone tracking,
IBTrACS best-track verification, persistence of verification products,
and optional generation of standard verification figures and
multi-lead tropical cyclone field sequences.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from aiweather.forecast import open_forecast
from aiweather.plotting import (
    plot_pressure_evolution,
    plot_tc_field_sequence,
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


def _center_at_lead(
    records,
    lead_time_hours: int | float,
):
    """
    Return a track center at one lead time.

    Parameters
    ----------
    records
        TrackRecord collection.

    lead_time_hours
        Requested forecast lead time.

    Returns
    -------
    tuple or None
        ``(latitude, longitude)`` when available.
    """
    for record in records:
        if (
            float(record.lead_time_hours)
            == float(lead_time_hours)
        ):
            return (
                float(record.latitude),
                float(record.longitude),
            )

    return None


def _generate_field_sequence_plot(
    *,
    forecast,
    tracking: TrackingWorkflowResult,
    verification: VerificationWorkflowResult,
    field_lead_times: list[int | float],
    plot_output_dir: Path,
) -> Path:
    """
    Generate a multi-lead tropical cyclone field sequence.
    """
    import matplotlib.pyplot as plt

    plot_output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    centers_by_lead = {}
    reference_centers = {}

    evaluation = tracking.evaluation

    wuduan_records = []
    vitart_records = []

    if (
        evaluation is not None
        and evaluation.wuduan_match is not None
    ):
        wuduan_records = (
            evaluation.wuduan_match.records
        )

    if (
        evaluation is not None
        and evaluation.vitart_match is not None
    ):
        vitart_records = (
            evaluation.vitart_match.records
        )

    observations = (
        verification.observations
    )

    for lead in field_lead_times:

        centers = {}

        native_center = _center_at_lead(
            tracking.native_records,
            lead,
        )

        wuduan_center = _center_at_lead(
            wuduan_records,
            lead,
        )

        vitart_center = _center_at_lead(
            vitart_records,
            lead,
        )

        ibtracs_center = _center_at_lead(
            observations,
            lead,
        )

        if native_center is not None:
            centers[
                "Native"
            ] = native_center

        if wuduan_center is not None:
            centers[
                "WuDuan"
            ] = wuduan_center

        if vitart_center is not None:
            centers[
                "Vitart"
            ] = vitart_center

        if ibtracs_center is not None:
            centers[
                "IBTrACS"
            ] = ibtracs_center

        centers_by_lead[
            lead
        ] = centers

        if ibtracs_center is not None:
            reference_centers[
                lead
            ] = ibtracs_center

        elif native_center is not None:
            reference_centers[
                lead
            ] = native_center

    figure, _ = plot_tc_field_sequence(
        forecast.dataset,
        lead_times=field_lead_times,
        centers_by_lead=(
            centers_by_lead
        ),
        reference_centers=(
            reference_centers
        ),
        latitude_margin=8.0,
        longitude_margin=12.0,
        quiver_stride=16,
        pressure_interval_hpa=4.0,
        ncols=2,
        figsize=(
            14.0,
            10.0,
        ),
    )

    figure.suptitle(
        "Tropical cyclone forecast fields",
        fontsize=14,
        y=0.98,
    )

    path = (
        plot_output_dir
        / "field_sequence.png"
    )

    figure.savefig(
        path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(
        figure
    )

    return path


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
    field_lead_times: list[
        int | float
    ] | None = None,
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

    field_lead_times
        Optional forecast lead times for generating a shared-scale
        tropical cyclone field sequence.

        Field-sequence generation only occurs when
        ``generate_plots=True``.

        Example::

            [54, 78, 96, 120]

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

    if field_lead_times is not None:

        if not isinstance(
            field_lead_times,
            list,
        ):
            raise TypeError(
                "field_lead_times must be "
                "a list or None."
            )

        if not field_lead_times:
            raise ValueError(
                "field_lead_times cannot "
                "be empty."
            )

        for lead in field_lead_times:
            if not isinstance(
                lead,
                (int, float),
            ):
                raise TypeError(
                    "field_lead_times must contain "
                    "numeric values."
                )

            if lead < 0:
                raise ValueError(
                    "field_lead_times cannot contain "
                    "negative values."
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

        if field_lead_times is not None:

            field_path = (
                _generate_field_sequence_plot(
                    forecast=forecast,
                    tracking=tracking,
                    verification=verification,
                    field_lead_times=(
                        field_lead_times
                    ),
                    plot_output_dir=(
                        resolved_plot_output_dir
                    ),
                )
            )

            plot_paths[
                "field_sequence"
            ] = field_path

    return TCVerificationPipelineResult(
        tracking=tracking,
        verification=verification,
        output_dir=exported,
        plot_paths=plot_paths,
    )