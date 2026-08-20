import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import pytest

from aiweather.plotting.lead_time import (
    plot_lead_time_metric,
    plot_lead_time_summary,
)


def make_table():
    return pd.DataFrame(
        {
            "model_name": [
                "graphcast",
                "graphcast",
                "graphcast",
                "graphcast",
            ],
            "tracker": [
                "native",
                "native",
                "wuduan",
                "wuduan",
            ],
            "lead_time_bin": [
                "72-120h",
                "48-72h",
                "72-120h",
                "48-72h",
            ],
            "lead_time_start_hours": [
                72.0,
                48.0,
                72.0,
                48.0,
            ],
            "lead_time_end_hours": [
                120.0,
                72.0,
                120.0,
                72.0,
            ],
            "rmse_track_error_km": [
                220.0,
                180.0,
                210.0,
                170.0,
            ],
            "pressure_rmse_pa": [
                3000.0,
                2000.0,
                2800.0,
                1800.0,
            ],
            "wind_rmse_ms": [
                25.0,
                18.0,
                23.0,
                16.0,
            ],
            "mean_track_error_km": [
                200.0,
                160.0,
                190.0,
                150.0,
            ],
            "median_track_error_km": [
                195.0,
                155.0,
                185.0,
                145.0,
            ],
            "pressure_mae_pa": [
                2500.0,
                1500.0,
                2300.0,
                1400.0,
            ],
            "wind_mae_ms": [
                22.0,
                15.0,
                20.0,
                14.0,
            ],
        }
    )


def test_plot_lead_time_metric():
    figure, ax = plot_lead_time_metric(
        make_table(),
        metric="rmse_track_error_km",
        title="Track RMSE",
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
        "Wuduan",
    ]

    assert ax.get_xlabel() == (
        "Forecast lead time"
    )

    assert ax.get_ylabel() == (
        "Track RMSE [km]"
    )

    figure.canvas.draw()

    plt.close(
        figure
    )


def test_plot_lead_time_sorts_bins():
    figure, ax = plot_lead_time_metric(
        make_table(),
        metric="rmse_track_error_km",
    )

    native = ax.lines[0]

    assert list(
        native.get_xdata()
    ) == [
        60.0,
        96.0,
    ]

    assert list(
        native.get_ydata()
    ) == [
        180.0,
        220.0,
    ]

    labels = [
        label.get_text()
        for label in ax.get_xticklabels()
    ]

    assert labels == [
        "48-72h",
        "72-120h",
    ]

    plt.close(
        figure
    )


def test_plot_pressure_converts_to_hpa():
    figure, ax = plot_lead_time_metric(
        make_table(),
        metric="pressure_rmse_pa",
    )

    native = ax.lines[0]

    assert list(
        native.get_ydata()
    ) == [
        20.0,
        30.0,
    ]

    assert ax.get_ylabel() == (
        "Pressure RMSE [hPa]"
    )

    plt.close(
        figure
    )


def test_plot_lead_time_from_csv(
    tmp_path,
):
    path = (
        tmp_path
        / "batch_lead_time.csv"
    )

    make_table().to_csv(
        path,
        index=False,
    )

    figure, ax = plot_lead_time_metric(
        path,
        metric="wind_rmse_ms",
    )

    assert len(ax.lines) == 2

    assert ax.get_ylabel() == (
        "Wind RMSE [m/s]"
    )

    plt.close(
        figure
    )


def test_plot_lead_time_invalid_data():
    with pytest.raises(
        TypeError,
        match="DataFrame or CSV path",
    ):
        plot_lead_time_metric(
            []
        )


def test_plot_lead_time_missing_columns():
    with pytest.raises(
        ValueError,
        match="missing required columns",
    ):
        plot_lead_time_metric(
            pd.DataFrame(
                {
                    "tracker": [
                        "native",
                    ],
                }
            )
        )


def test_plot_lead_time_invalid_metric():
    with pytest.raises(
        ValueError,
        match="Unknown lead-time metric",
    ):
        plot_lead_time_metric(
            make_table(),
            metric="not_a_metric",
        )


def test_plot_lead_time_summary():
    figure, axes = (
        plot_lead_time_summary(
            make_table(),
            title=(
                "GraphCast lead-time "
                "verification"
            ),
        )
    )

    assert figure is not None

    assert len(
        axes
    ) == 3

    assert axes[0].get_ylabel() == (
        "Track RMSE [km]"
    )

    assert axes[1].get_ylabel() == (
        "Pressure RMSE [hPa]"
    )

    assert axes[2].get_ylabel() == (
        "Wind RMSE [m/s]"
    )

    figure.canvas.draw()

    plt.close(
        figure
    )

def test_plot_lead_time_global_bin_order():
    figure, ax = plot_lead_time_metric(
        make_table(),
        metric="rmse_track_error_km",
    )

    labels = [
        label.get_text()
        for label in ax.get_xticklabels()
    ]

    assert labels == [
        "48-72h",
        "72-120h",
    ]

    plt.close(
        figure
    )

def test_plot_lead_time_global_bin_order_with_missing_bins():
    table = make_table()

    early = pd.DataFrame(
        {
            "model_name": [
                "graphcast",
                "graphcast",
            ],
            "tracker": [
                "wuduan",
                "wuduan",
            ],
            "lead_time_bin": [
                "0-24h",
                "24-48h",
            ],
            "lead_time_start_hours": [
                0.0,
                24.0,
            ],
            "lead_time_end_hours": [
                24.0,
                48.0,
            ],
            "rmse_track_error_km": [
                100.0,
                130.0,
            ],
            "pressure_rmse_pa": [
                500.0,
                800.0,
            ],
            "wind_rmse_ms": [
                5.0,
                8.0,
            ],
            "mean_track_error_km": [
                90.0,
                120.0,
            ],
            "median_track_error_km": [
                85.0,
                115.0,
            ],
            "pressure_mae_pa": [
                400.0,
                700.0,
            ],
            "wind_mae_ms": [
                4.0,
                7.0,
            ],
        }
    )

    table = pd.concat(
        [
            table,
            early,
        ],
        ignore_index=True,
    )

    figure, ax = plot_lead_time_metric(
        table,
        metric="rmse_track_error_km",
    )

    labels = [
        label.get_text()
        for label in ax.get_xticklabels()
    ]

    assert labels == [
        "0-24h",
        "24-48h",
        "48-72h",
        "72-120h",
    ]

    plt.close(
        figure
    )