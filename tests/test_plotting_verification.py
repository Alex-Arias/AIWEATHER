import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import pytest

from aiweather.plotting.verification import (
    plot_pressure_evolution,
    plot_track_error,
    plot_wind_evolution,
)


def make_table():
    return pd.DataFrame(
        {
            "lead_time_hours": [
                54,
                60,
                66,
            ],
            "track_error_km": [
                100.0,
                150.0,
                200.0,
            ],
            "forecast_pressure_pa": [
                100000.0,
                99500.0,
                99000.0,
            ],
            "observed_pressure_pa": [
                98000.0,
                97000.0,
                96000.0,
            ],
            "forecast_wind_ms": [
                20.0,
                22.0,
                24.0,
            ],
            "observed_wind_ms": [
                40.0,
                45.0,
                50.0,
            ],
        }
    )


def test_plot_track_error():
    tables = {
        "Native": make_table(),
        "WuDuan": make_table(),
    }

    figure, ax = plot_track_error(
        tables,
        title="Track error",
    )

    assert figure is not None
    assert ax is not None

    assert len(ax.lines) == 2

    labels = [
        line.get_label()
        for line in ax.lines
    ]

    assert labels == [
        "Native",
        "WuDuan",
    ]

    assert ax.get_xlabel() == (
        "Forecast lead time [h]"
    )

    assert ax.get_ylabel() == (
        "Track error [km]"
    )

    figure.canvas.draw()

    plt.close(
        figure
    )


def test_plot_pressure_evolution():
    tables = {
        "Native": make_table(),
        "WuDuan": make_table(),
    }

    figure, ax = plot_pressure_evolution(
        tables,
        title="Pressure",
    )

    assert figure is not None
    assert ax is not None

    # Native + IBTrACS + WuDuan
    assert len(ax.lines) == 3

    labels = [
        line.get_label()
        for line in ax.lines
    ]

    assert labels == [
        "Native",
        "IBTrACS",
        "WuDuan",
    ]

    assert ax.get_ylabel() == (
        "Minimum pressure [hPa]"
    )

    observed = ax.lines[
        1
    ].get_ydata()

    assert list(observed) == [
        980.0,
        970.0,
        960.0,
    ]

    figure.canvas.draw()

    plt.close(
        figure
    )


def test_plot_wind_evolution():
    tables = {
        "Native": make_table(),
        "WuDuan": make_table(),
    }

    figure, ax = plot_wind_evolution(
        tables,
        title="Maximum wind",
    )

    assert figure is not None
    assert ax is not None

    assert len(ax.lines) == 3

    labels = [
        line.get_label()
        for line in ax.lines
    ]

    assert labels == [
        "Native",
        "IBTrACS",
        "WuDuan",
    ]

    assert ax.get_ylabel() == (
        "Maximum wind [m/s]"
    )

    figure.canvas.draw()

    plt.close(
        figure
    )


def test_plot_track_error_empty():
    figure, ax = plot_track_error(
        {}
    )

    assert len(ax.lines) == 0

    figure.canvas.draw()

    plt.close(
        figure
    )


def test_plot_pressure_evolution_empty():
    figure, ax = plot_pressure_evolution(
        {}
    )

    assert len(ax.lines) == 0

    figure.canvas.draw()

    plt.close(
        figure
    )


def test_plot_wind_evolution_empty():
    figure, ax = plot_wind_evolution(
        {}
    )

    assert len(ax.lines) == 0

    figure.canvas.draw()

    plt.close(
        figure
    )


def test_plotting_verification_invalid_mapping():
    with pytest.raises(
        TypeError,
        match="mapping",
    ):
        plot_track_error(
            []
        )


def test_plotting_verification_invalid_table():
    with pytest.raises(
        TypeError,
        match="DataFrame",
    ):
        plot_track_error(
            {
                "Native": [],
            }
        )