import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest
import xarray as xr

from aiweather.plotting.fields import (
    plot_tc_field,
    wind_speed,
)


def make_dataset():
    lat = np.array(
        [
            10.0,
            11.0,
            12.0,
        ]
    )

    lon = np.array(
        [
            250.0,
            251.0,
            252.0,
            253.0,
        ]
    )

    lead_time = np.array(
        [
            0,
            6,
            12,
        ]
    )

    shape = (
        len(lead_time),
        len(lat),
        len(lon),
    )

    u10m = np.full(
        shape,
        3.0,
        dtype=float,
    )

    v10m = np.full(
        shape,
        4.0,
        dtype=float,
    )

    msl = np.full(
        shape,
        100000.0,
        dtype=float,
    )

    # Add a small pressure gradient so contouring has
    # multiple usable levels.
    for index in range(
        len(lon)
    ):
        msl[
            :,
            :,
            index,
        ] += index * 200.0

    return xr.Dataset(
        {
            "u10m": (
                (
                    "lead_time",
                    "lat",
                    "lon",
                ),
                u10m,
            ),
            "v10m": (
                (
                    "lead_time",
                    "lat",
                    "lon",
                ),
                v10m,
            ),
            "msl": (
                (
                    "lead_time",
                    "lat",
                    "lon",
                ),
                msl,
            ),
        },
        coords={
            "lead_time": lead_time,
            "lat": lat,
            "lon": lon,
        },
    )


def test_wind_speed():
    result = wind_speed(
        np.array(
            [
                3.0,
                5.0,
            ]
        ),
        np.array(
            [
                4.0,
                12.0,
            ]
        ),
    )

    np.testing.assert_allclose(
        result,
        [
            5.0,
            13.0,
        ],
    )


def test_plot_tc_field():
    dataset = make_dataset()

    figure, ax = plot_tc_field(
        dataset,
        lead_time_hours=6,
        quiver_stride=1,
        pressure_interval_hpa=2.0,
        title="Synthetic TC field",
    )

    assert figure is not None
    assert ax is not None

    assert ax.get_title() == (
        "Synthetic TC field"
    )

    figure.canvas.draw()

    plt.close(
        figure
    )


def test_plot_tc_field_missing_variable():
    dataset = make_dataset().drop_vars(
        "msl"
    )

    with pytest.raises(
        ValueError,
        match="missing required variables",
    ):
        plot_tc_field(
            dataset,
            lead_time_hours=6,
        )


def test_plot_tc_field_invalid_dataset():
    with pytest.raises(
        TypeError,
        match="xarray Dataset",
    ):
        plot_tc_field(
            "invalid",
            lead_time_hours=6,
        )


def test_plot_tc_field_invalid_quiver_stride():
    dataset = make_dataset()

    with pytest.raises(
        ValueError,
        match="at least 1",
    ):
        plot_tc_field(
            dataset,
            lead_time_hours=6,
            quiver_stride=0,
        )


def test_plot_tc_field_invalid_pressure_interval():
    dataset = make_dataset()

    with pytest.raises(
        ValueError,
        match="must be positive",
    ):
        plot_tc_field(
            dataset,
            lead_time_hours=6,
            pressure_interval_hpa=0.0,
        )


def test_plot_tc_field_outside_lead_range():
    dataset = make_dataset()

    with pytest.raises(
        IndexError,
        match="outside",
    ):
        plot_tc_field(
            dataset,
            lead_time_hours=240,
        )

def test_plot_tc_field_centers():
    dataset = make_dataset()

    figure, ax = plot_tc_field(
        dataset,
        lead_time_hours=6,
        centers={
            "Native": (
                11.0,
                251.0,
            ),
            "IBTrACS": (
                10.5,
                -109.0,
            ),
        },
    )

    labels = [
        collection.get_label()
        for collection in ax.collections
    ]

    assert "Native" in labels
    assert "IBTrACS" in labels

    figure.canvas.draw()

    plt.close(
        figure
    )


def test_plot_tc_field_invalid_centers():
    dataset = make_dataset()

    with pytest.raises(
        TypeError,
        match="mapping",
    ):
        plot_tc_field(
            dataset,
            lead_time_hours=6,
            centers=[],
        )


def test_plot_tc_field_invalid_center_value():
    dataset = make_dataset()

    with pytest.raises(
        TypeError,
        match="latitude, longitude",
    ):
        plot_tc_field(
            dataset,
            lead_time_hours=6,
            centers={
                "Native": 10.0,
            },
        )