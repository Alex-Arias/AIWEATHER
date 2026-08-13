import numpy as np
import pandas as pd
import pytest

from aiweather.verification.ibtracs import (
    HPA_TO_PA,
    KNOT_TO_MS,
    read_ibtracs_csv,
)


def write_test_csv(path):
    data = pd.DataFrame(
        {
            "SID": [
                "2026200N10000",
                "2026200N10000",
                "OTHER",
            ],
            "ISO_TIME": [
                "2026-07-24 06:00:00",
                "2026-07-24 00:00:00",
                "2026-07-24 00:00:00",
            ],
            "LAT": [
                11.0,
                10.0,
                30.0,
            ],
            "LON": [
                -111.0,
                -110.0,
                140.0,
            ],
            "USA_PRES": [
                995.0,
                1000.0,
                980.0,
            ],
            "USA_WIND": [
                40.0,
                35.0,
                80.0,
            ],
        }
    )

    data.to_csv(
        path,
        index=False,
    )


def test_read_ibtracs_csv(tmp_path):
    path = (
        tmp_path
        / "ibtracs.csv"
    )

    write_test_csv(path)

    points = read_ibtracs_csv(
        path,
        sid="2026200N10000",
    )

    assert len(points) == 2

    assert points[0].valid_time == np.datetime64(
        "2026-07-24T00:00:00"
    )

    assert points[1].valid_time == np.datetime64(
        "2026-07-24T06:00:00"
    )

    assert points[0].latitude == pytest.approx(
        10.0
    )

    assert points[0].longitude == pytest.approx(
        -110.0
    )

    assert points[0].pressure == pytest.approx(
        1000.0 * HPA_TO_PA
    )

    assert points[0].max_wind == pytest.approx(
        35.0 * KNOT_TO_MS
    )

    assert points[0].pressure_units == "Pa"
    assert points[0].wind_units == "m/s"


def test_read_ibtracs_csv_missing_intensity(
    tmp_path,
):
    path = (
        tmp_path
        / "ibtracs.csv"
    )

    data = pd.DataFrame(
        {
            "SID": ["TEST"],
            "ISO_TIME": [
                "2026-07-24 00:00:00"
            ],
            "LAT": [10.0],
            "LON": [-110.0],
            "USA_PRES": [np.nan],
            "USA_WIND": [np.nan],
        }
    )

    data.to_csv(
        path,
        index=False,
    )

    points = read_ibtracs_csv(
        path,
        sid="TEST",
    )

    assert len(points) == 1
    assert points[0].pressure is None
    assert points[0].max_wind is None
    assert points[0].pressure_units is None
    assert points[0].wind_units is None


def test_read_ibtracs_csv_unknown_sid(
    tmp_path,
):
    path = (
        tmp_path
        / "ibtracs.csv"
    )

    write_test_csv(path)

    with pytest.raises(
        ValueError,
        match="was not found",
    ):
        read_ibtracs_csv(
            path,
            sid="UNKNOWN",
        )


def test_read_ibtracs_csv_missing_file():
    with pytest.raises(
        FileNotFoundError,
    ):
        read_ibtracs_csv(
            "missing.csv",
            sid="TEST",
        )


def test_read_ibtracs_csv_missing_columns(
    tmp_path,
):
    path = (
        tmp_path
        / "ibtracs.csv"
    )

    pd.DataFrame(
        {
            "SID": ["TEST"],
            "ISO_TIME": [
                "2026-07-24 00:00:00"
            ],
        }
    ).to_csv(
        path,
        index=False,
    )

    with pytest.raises(
        ValueError,
        match="missing required columns",
    ):
        read_ibtracs_csv(
            path,
            sid="TEST",
        )


def test_read_ibtracs_csv_empty_sid(
    tmp_path,
):
    path = (
        tmp_path
        / "ibtracs.csv"
    )

    write_test_csv(path)

    with pytest.raises(
        ValueError,
        match="sid cannot be empty",
    ):
        read_ibtracs_csv(
            path,
            sid="   ",
        )