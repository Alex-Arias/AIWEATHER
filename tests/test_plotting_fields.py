import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest
import xarray as xr

from aiweather.plotting.fields import (
    plot_tc_field,
    plot_tc_field_sequence,
    storm_centered_extent,
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

def test_storm_centered_extent():
    from aiweather.plotting.fields import (
        storm_centered_extent,
    )

    extent = storm_centered_extent(
        15.0,
        250.0,
        latitude_margin=5.0,
        longitude_margin=10.0,
    )

    assert extent == pytest.approx(
        (
            -120.0,
            -100.0,
            10.0,
            20.0,
        )
    )


def test_storm_centered_extent_invalid_latitude_margin():
    from aiweather.plotting.fields import (
        storm_centered_extent,
    )

    with pytest.raises(
        ValueError,
        match="latitude_margin",
    ):
        storm_centered_extent(
            15.0,
            250.0,
            latitude_margin=0.0,
        )


def test_storm_centered_extent_invalid_longitude_margin():
    from aiweather.plotting.fields import (
        storm_centered_extent,
    )

    with pytest.raises(
        ValueError,
        match="longitude_margin",
    ):
        storm_centered_extent(
            15.0,
            250.0,
            longitude_margin=0.0,
        )

def test_plot_tc_field_sequence():
    dataset = make_dataset()

    figure, axes = plot_tc_field_sequence(
        dataset,
        lead_times=[
            0,
            6,
            12,
            6,
        ],
        centers_by_lead={
            0: {
                "Native": (
                    11.0,
                    251.0,
                ),
                "IBTrACS": (
                    10.5,
                    -109.0,
                ),
            },
            6: {
                "Native": (
                    11.5,
                    251.5,
                ),
            },
            12: {
                "Native": (
                    12.0,
                    252.0,
                ),
            },
        },
        ncols=2,
        quiver_stride=1,
        pressure_interval_hpa=2.0,
    )

    assert axes.shape == (
        2,
        2,
    )

    titles = [
        ax.get_title()
        for ax in axes.ravel()
    ]

    assert titles == [
        "+0 h",
        "+6 h",
        "+12 h",
        "+6 h",
    ]

    figure.canvas.draw()

    plt.close(
        figure
    )


def test_plot_tc_field_sequence_hides_unused_axis():
    dataset = make_dataset()

    figure, axes = plot_tc_field_sequence(
        dataset,
        lead_times=[
            0,
            6,
            12,
        ],
        ncols=2,
    )

    assert axes.shape == (
        2,
        2,
    )

    assert (
        axes.ravel()[-1].get_visible()
        is False
    )

    figure.canvas.draw()

    plt.close(
        figure
    )


def test_plot_tc_field_sequence_empty_leads():
    dataset = make_dataset()

    with pytest.raises(
        ValueError,
        match="cannot be empty",
    ):
        plot_tc_field_sequence(
            dataset,
            lead_times=[],
        )


def test_plot_tc_field_sequence_invalid_leads():
    dataset = make_dataset()

    with pytest.raises(
        TypeError,
        match="lead_times",
    ):
        plot_tc_field_sequence(
            dataset,
            lead_times=(0, 6),
        )


def test_plot_tc_field_sequence_invalid_ncols():
    dataset = make_dataset()

    with pytest.raises(
        ValueError,
        match="at least 1",
    ):
        plot_tc_field_sequence(
            dataset,
            lead_times=[
                0,
            ],
            ncols=0,
        )


def test_plot_tc_field_sequence_invalid_centers_mapping():
    dataset = make_dataset()

    with pytest.raises(
        TypeError,
        match="centers_by_lead",
    ):
        plot_tc_field_sequence(
            dataset,
            lead_times=[
                0,
            ],
            centers_by_lead=[],
        )


def test_plot_tc_field_sequence_invalid_reference_mapping():
    dataset = make_dataset()

    with pytest.raises(
        TypeError,
        match="reference_centers",
    ):
        plot_tc_field_sequence(
            dataset,
            lead_times=[
                0,
            ],
            reference_centers=[],
        )

def test_plot_tc_field_without_colorbar():
    dataset = make_dataset()

    figure, ax = plot_tc_field(
        dataset,
        lead_time_hours=6,
        add_colorbar=False,
    )

    # Only the main plotting axes should exist.
    assert len(
        figure.axes
    ) == 1

    figure.canvas.draw()

    plt.close(
        figure
    )


def test_plot_tc_field_with_wind_speed_limits():
    dataset = make_dataset()

    figure, ax = plot_tc_field(
        dataset,
        lead_time_hours=6,
        wind_speed_limits=(
            0.0,
            10.0,
        ),
    )

    # pcolormesh is the first collection.
    shading = ax.collections[
        0
    ]

    assert shading.norm.vmin == pytest.approx(
        0.0
    )

    assert shading.norm.vmax == pytest.approx(
        10.0
    )

    figure.canvas.draw()

    plt.close(
        figure
    )


def test_plot_tc_field_invalid_wind_limits_type():
    dataset = make_dataset()

    with pytest.raises(
        TypeError,
        match="wind_speed_limits",
    ):
        plot_tc_field(
            dataset,
            lead_time_hours=6,
            wind_speed_limits=[
                0.0,
                10.0,
            ],
        )


def test_plot_tc_field_invalid_wind_limits_order():
    dataset = make_dataset()

    with pytest.raises(
        ValueError,
        match="greater than minimum",
    ):
        plot_tc_field(
            dataset,
            lead_time_hours=6,
            wind_speed_limits=(
                10.0,
                5.0,
            ),
        )


def test_plot_tc_field_invalid_add_colorbar():
    dataset = make_dataset()

    with pytest.raises(
        TypeError,
        match="boolean",
    ):
        plot_tc_field(
            dataset,
            lead_time_hours=6,
            add_colorbar="yes",
        )


def test_plot_tc_field_sequence_shared_colorbar():
    dataset = make_dataset()

    figure, axes = plot_tc_field_sequence(
        dataset,
        lead_times=[
            0,
            6,
            12,
            6,
        ],
        ncols=2,
        quiver_stride=1,
        pressure_interval_hpa=2.0,
    )

    # Four panel axes plus exactly one shared
    # colorbar axes.
    assert len(
        figure.axes
    ) == 5

    panel_norms = []

    for ax in axes.ravel():
        shading = ax.collections[
            0
        ]

        panel_norms.append(
            (
                shading.norm.vmin,
                shading.norm.vmax,
            )
        )

    assert all(
        limits
        == panel_norms[0]
        for limits in panel_norms
    )

    assert panel_norms[
        0
    ][0] == pytest.approx(
        0.0
    )

    figure.canvas.draw()

    plt.close(
        figure
    )