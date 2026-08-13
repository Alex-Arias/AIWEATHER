import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest

from aiweather.plotting.tracks import (
    normalize_longitude,
    plot_track_map,
)
from aiweather.tracking.records import (
    TrackRecord,
)


def make_record(
    lead_time_hours,
    latitude,
    longitude,
):
    return TrackRecord(
        lead_time_hours=lead_time_hours,
        valid_time=(
            np.datetime64(
                "2026-07-24T00:00:00"
            )
            + np.timedelta64(
                lead_time_hours,
                "h",
            )
        ),
        latitude=latitude,
        longitude=longitude,
        pressure=100000.0,
        pressure_units="Pa",
        max_wind=20.0,
        wind_units="m/s",
    )


def test_normalize_longitude_scalar():
    assert normalize_longitude(
        254.0
    ) == pytest.approx(
        -106.0
    )

    assert normalize_longitude(
        -107.6
    ) == pytest.approx(
        -107.6
    )


def test_normalize_longitude_array():
    result = normalize_longitude(
        [
            254.0,
            180.0,
            360.0,
            -107.6,
        ]
    )

    expected = np.array(
        [
            -106.0,
            -180.0,
            0.0,
            -107.6,
        ]
    )

    np.testing.assert_allclose(
        result,
        expected,
    )


def test_plot_track_map():
    observations = [
        make_record(
            54,
            12.6,
            -107.6,
        ),
        make_record(
            60,
            13.2,
            -108.8,
        ),
    ]

    native = [
        make_record(
            54,
            13.0,
            254.0,
        ),
        make_record(
            60,
            13.5,
            253.0,
        ),
    ]

    figure, ax = plot_track_map(
        observations,
        forecasts={
            "Native": native,
        },
        title="Genevieve",
    )

    assert figure is not None
    assert ax is not None

    assert len(
        ax.lines
    ) == 2

    labels = [
        line.get_label()
        for line in ax.lines
    ]

    assert labels == [
        "IBTrACS",
        "Native",
    ]

    native_x = ax.lines[
        1
    ].get_xdata()

    np.testing.assert_allclose(
        native_x,
        [
            -106.0,
            -107.0,
        ],
    )

    figure.canvas.draw()

    plt.close(
        figure
    )


def test_plot_track_map_multiple_forecasts():
    observations = [
        make_record(
            54,
            12.6,
            -107.6,
        ),
    ]

    forecasts = {
        "Native": [
            make_record(
                54,
                13.0,
                254.0,
            )
        ],
        "WuDuan": [
            make_record(
                54,
                12.9,
                253.9,
            )
        ],
        "Vitart": [
            make_record(
                96,
                16.75,
                248.75,
            )
        ],
    }

    figure, ax = plot_track_map(
        observations,
        forecasts=forecasts,
    )

    assert len(
        ax.lines
    ) == 4

    labels = {
        line.get_label()
        for line in ax.lines
    }

    assert labels == {
        "IBTrACS",
        "Native",
        "WuDuan",
        "Vitart",
    }

    figure.canvas.draw()

    plt.close(
        figure
    )


def test_plot_track_map_annotations():
    observations = []

    native = [
        make_record(
            24,
            10.0,
            250.0,
        ),
        make_record(
            30,
            11.0,
            249.0,
        ),
        make_record(
            48,
            12.0,
            248.0,
        ),
    ]

    figure, ax = plot_track_map(
        observations,
        forecasts={
            "Native": native,
        },
        annotate_lead_time=True,
        lead_time_interval_hours=24,
    )

    labels = [
        text.get_text()
        for text in ax.texts
    ]

    assert labels == [
        "+24h",
        "+48h",
    ]

    plt.close(
        figure
    )


def test_plot_track_map_invalid_forecasts():
    with pytest.raises(
        TypeError,
        match="mapping",
    ):
        plot_track_map(
            [],
            forecasts=[],
        )


def test_plot_track_map_invalid_interval():
    with pytest.raises(
        ValueError,
        match="greater than zero",
    ):
        plot_track_map(
            [],
            lead_time_interval_hours=0,
        )