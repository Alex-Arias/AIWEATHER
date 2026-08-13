"""
Tropical cyclone verification plotting utilities.

Provides plots for forecast track error and intensity evolution using
AIWeather verification tables.
"""

from __future__ import annotations

from collections.abc import Mapping

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure


def _validate_tables(
    verifications: Mapping[str, pd.DataFrame],
) -> None:
    """
    Validate verification table mapping.
    """
    if not isinstance(
        verifications,
        Mapping,
    ):
        raise TypeError(
            "verifications must be a mapping."
        )

    for name, table in verifications.items():
        if not isinstance(
            name,
            str,
        ):
            raise TypeError(
                "verification names must be strings."
            )

        if not isinstance(
            table,
            pd.DataFrame,
        ):
            raise TypeError(
                f"verifications[{name!r}] must be "
                "a pandas DataFrame."
            )


def plot_track_error(
    verifications: Mapping[
        str,
        pd.DataFrame,
    ],
    *,
    ax: Axes | None = None,
    title: str | None = None,
) -> tuple[
    Figure,
    Axes,
]:
    """
    Plot tropical cyclone track error versus forecast lead time.
    """
    _validate_tables(
        verifications
    )

    if ax is None:
        figure, ax = plt.subplots(
            figsize=(9, 5),
        )
    else:
        figure = ax.figure

    for name, table in verifications.items():
        if table.empty:
            continue

        ax.plot(
            table["lead_time_hours"],
            table["track_error_km"],
            marker="o",
            linewidth=1.6,
            label=name,
        )

    ax.set_xlabel(
        "Forecast lead time [h]"
    )

    ax.set_ylabel(
        "Track error [km]"
    )

    if title is not None:
        ax.set_title(
            title
        )

    ax.grid(
        True,
        alpha=0.3,
    )

    if verifications:
        ax.legend()

    return (
        figure,
        ax,
    )


def plot_pressure_evolution(
    verifications: Mapping[
        str,
        pd.DataFrame,
    ],
    *,
    ax: Axes | None = None,
    title: str | None = None,
) -> tuple[
    Figure,
    Axes,
]:
    """
    Plot forecast and observed minimum central pressure.
    """
    _validate_tables(
        verifications
    )

    if ax is None:
        figure, ax = plt.subplots(
            figsize=(9, 5),
        )
    else:
        figure = ax.figure

    observed_plotted = False

    for name, table in verifications.items():
        if table.empty:
            continue

        lead = table[
            "lead_time_hours"
        ]

        forecast_pressure = (
            table[
                "forecast_pressure_pa"
            ]
            / 100.0
        )

        ax.plot(
            lead,
            forecast_pressure,
            marker="o",
            linewidth=1.6,
            label=name,
        )

        if not observed_plotted:
            observed_pressure = (
                table[
                    "observed_pressure_pa"
                ]
                / 100.0
            )

            ax.plot(
                lead,
                observed_pressure,
                marker="o",
                linewidth=2.0,
                label="IBTrACS",
            )

            observed_plotted = True

    ax.set_xlabel(
        "Forecast lead time [h]"
    )

    ax.set_ylabel(
        "Minimum pressure [hPa]"
    )

    if title is not None:
        ax.set_title(
            title
        )

    ax.grid(
        True,
        alpha=0.3,
    )

    if verifications:
        ax.legend()

    return (
        figure,
        ax,
    )


def plot_wind_evolution(
    verifications: Mapping[
        str,
        pd.DataFrame,
    ],
    *,
    ax: Axes | None = None,
    title: str | None = None,
) -> tuple[
    Figure,
    Axes,
]:
    """
    Plot forecast and observed maximum sustained wind.
    """
    _validate_tables(
        verifications
    )

    if ax is None:
        figure, ax = plt.subplots(
            figsize=(9, 5),
        )
    else:
        figure = ax.figure

    observed_plotted = False

    for name, table in verifications.items():
        if table.empty:
            continue

        lead = table[
            "lead_time_hours"
        ]

        forecast_wind = table[
            "forecast_wind_ms"
        ]

        ax.plot(
            lead,
            forecast_wind,
            marker="o",
            linewidth=1.6,
            label=name,
        )

        if not observed_plotted:
            observed_wind = table[
                "observed_wind_ms"
            ]

            ax.plot(
                lead,
                observed_wind,
                marker="o",
                linewidth=2.0,
                label="IBTrACS",
            )

            observed_plotted = True

    ax.set_xlabel(
        "Forecast lead time [h]"
    )

    ax.set_ylabel(
        "Maximum wind [m/s]"
    )

    if title is not None:
        ax.set_title(
            title
        )

    ax.grid(
        True,
        alpha=0.3,
    )

    if verifications:
        ax.legend()

    return (
        figure,
        ax,
    )