"""
Lead-time verification plotting utilities.

Provides plots for QC-filtered batch tropical-cyclone
verification statistics grouped by forecast lead-time bin.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure


_REQUIRED_COLUMNS = {
    "model_name",
    "tracker",
    "lead_time_bin",
    "lead_time_start_hours",
    "lead_time_end_hours",
}


_METRICS = {
    "mean_track_error_km": (
        "Mean track error [km]",
        1.0,
    ),
    "rmse_track_error_km": (
        "Track RMSE [km]",
        1.0,
    ),
    "median_track_error_km": (
        "Median track error [km]",
        1.0,
    ),
    "pressure_mae_pa": (
        "Pressure MAE [hPa]",
        0.01,
    ),
    "pressure_rmse_pa": (
        "Pressure RMSE [hPa]",
        0.01,
    ),
    "wind_mae_ms": (
        "Wind MAE [m/s]",
        1.0,
    ),
    "wind_rmse_ms": (
        "Wind RMSE [m/s]",
        1.0,
    ),
}


def _load_lead_time_data(
    data: pd.DataFrame | str | Path,
) -> pd.DataFrame:
    """
    Load and validate lead-time verification statistics.
    """

    if isinstance(
        data,
        pd.DataFrame,
    ):
        table = data.copy()

    elif isinstance(
        data,
        (str, Path),
    ):
        table = pd.read_csv(
            data
        )

    else:
        raise TypeError(
            "data must be a pandas DataFrame "
            "or CSV path."
        )

    missing = (
        _REQUIRED_COLUMNS
        - set(table.columns)
    )

    if missing:
        raise ValueError(
            "lead-time table is missing "
            "required columns: "
            + ", ".join(
                sorted(missing)
            )
        )

    return table


def _series_label(
    *,
    model_name: str,
    tracker: str,
    multi_model: bool,
) -> str:
    """
    Build a readable plotted-series label.
    """

    tracker_label = (
        str(tracker)
        .replace("_", " ")
        .title()
    )

    if not multi_model:
        return tracker_label

    model_label = (
        str(model_name)
        .replace("_", " ")
        .title()
    )

    return (
        f"{model_label} / "
        f"{tracker_label}"
    )


def plot_lead_time_metric(
    data: pd.DataFrame | str | Path,
    *,
    metric: str = "rmse_track_error_km",
    ax: Axes | None = None,
    title: str | None = None,
) -> tuple[
    Figure,
    Axes,
]:
    """
    Plot one verification metric by forecast lead-time bin.

    Parameters
    ----------
    data
        Lead-time verification DataFrame or path to
        ``batch_lead_time.csv``.

    metric
        Verification metric to plot.

    ax
        Optional Matplotlib axis.

    title
        Optional figure title.

    Returns
    -------
    tuple
        Matplotlib Figure and Axes.
    """

    table = _load_lead_time_data(
        data
    )

    if metric not in _METRICS:
        raise ValueError(
            f"Unknown lead-time metric "
            f"{metric!r}. "
            f"Available metrics: "
            f"{', '.join(_METRICS)}"
        )

    if metric not in table.columns:
        raise ValueError(
            f"lead-time table does not "
            f"contain metric {metric!r}."
        )

    if ax is None:
        figure, ax = plt.subplots(
            figsize=(9, 5),
        )
    else:
        figure = ax.figure

    ylabel, scale = (
        _METRICS[
            metric
        ]
    )

    bins = (
        table[
            [
                "lead_time_bin",
                "lead_time_start_hours",
                "lead_time_end_hours",
            ]
        ]
        .drop_duplicates()
        .sort_values(
            "lead_time_start_hours"
        )
    )

    tick_positions = (
        bins["lead_time_start_hours"]
        + bins["lead_time_end_hours"]
    ) / 2.0

    model_count = (
        table[
            "model_name"
        ]
        .dropna()
        .nunique()
    )

    multi_model = (
        model_count > 1
    )

    grouped = table.groupby(
        [
            "model_name",
            "tracker",
        ],
        sort=True,
        dropna=False,
    )

    for (
        model_name,
        tracker,
    ), group in grouped:

        if group.empty:
            continue

        group = (
            group
            .sort_values(
                "lead_time_start_hours"
            )
        )

        values = (
            group[
                metric
            ]
            * scale
        )

        label = _series_label(
            model_name=model_name,
            tracker=tracker,
            multi_model=multi_model,
        )

        x = (
            group["lead_time_start_hours"]
            + group["lead_time_end_hours"]
        ) / 2.0

        ax.plot(
            x,
           values,
            marker="o",
            linewidth=1.6,
            label=label,
        )

    ax.set_xticks(
        tick_positions
    )

    ax.set_xticklabels(
        bins["lead_time_bin"]
    )

    ax.set_xlabel(
        "Forecast lead time"
    )

    ax.set_ylabel(
        ylabel
    )

    if title is not None:
        ax.set_title(
            title
        )

    ax.grid(
        True,
        alpha=0.3,
    )

    if not table.empty:
        ax.legend()

    return (
        figure,
        ax,
    )


def plot_lead_time_summary(
    data: pd.DataFrame | str | Path,
    *,
    title: str | None = None,
) -> tuple[
    Figure,
    object,
]:
    """
    Plot track, pressure, and wind RMSE by lead time.

    Returns a three-panel summary figure containing:

    - track RMSE [km]
    - pressure RMSE [hPa]
    - wind RMSE [m/s]
    """

    table = _load_lead_time_data(
        data
    )

    figure, axes = plt.subplots(
        3,
        1,
        figsize=(9, 12),
        sharex=True,
    )

    plot_lead_time_metric(
        table,
        metric=(
            "rmse_track_error_km"
        ),
        ax=axes[0],
        title="Track error",
    )

    plot_lead_time_metric(
        table,
        metric=(
            "pressure_rmse_pa"
        ),
        ax=axes[1],
        title="Central pressure error",
    )

    plot_lead_time_metric(
        table,
        metric=(
            "wind_rmse_ms"
        ),
        ax=axes[2],
        title="Maximum wind error",
    )

    if title is not None:
        figure.suptitle(
            title
        )

    figure.tight_layout()

    return (
        figure,
        axes,
    )