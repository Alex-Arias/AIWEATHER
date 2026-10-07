from pathlib import Path

import pandas as pd
import pytest

from scripts.build_operational_intensity_guidance import (
    availability_status,
    build_intensity_guidance,
)


def write_track(
    path: Path,
    *,
    leads,
    pressures,
    winds,
):
    times = (
        pd.Timestamp("2026-09-28 12:00:00")
        + pd.to_timedelta(leads, unit="h")
    )

    df = pd.DataFrame(
        {
            "lead_time_hours": leads,
            "valid_time": times,
            "latitude": [20.0] * len(leads),
            "longitude": [-120.0] * len(leads),
            "pressure": pressures,
            "pressure_units": ["Pa"] * len(leads),
            "max_wind": winds,
            "wind_units": ["m/s"] * len(leads),
        }
    )

    df.to_csv(path, index=False)


def test_availability_status():
    assert availability_status(3) == "SUPPORTED"
    assert availability_status(2) == "LIMITED"
    assert availability_status(1) == "INSUFFICIENT"


def test_build_guidance_retains_limited_tail(tmp_path):
    aifs2 = tmp_path / "aifs2.csv"
    graphcast = tmp_path / "graphcast.csv"
    pangu = tmp_path / "pangu.csv"

    write_track(
        aifs2,
        leads=[0, 6, 12],
        pressures=[99000, 99500, 100000],
        winds=[30, 25, 20],
    )

    write_track(
        graphcast,
        leads=[0, 6, 12],
        pressures=[99200, 99600, 100100],
        winds=[29, 24, 19],
    )

    write_track(
        pangu,
        leads=[0, 6],
        pressures=[99100, 99400],
        winds=[31, 26],
    )

    result = build_intensity_guidance(
        aifs2_path=aifs2,
        graphcast_path=graphcast,
        pangu_path=pangu,
        storm="Test Storm",
    )

    assert len(result) == 3

    assert list(result["n_models"]) == [
        3,
        3,
        2,
    ]

    assert list(result["availability_status"]) == [
        "SUPPORTED",
        "SUPPORTED",
        "LIMITED",
    ]

    assert list(result["intensity_supported"]) == [
        True,
        True,
        False,
    ]


def test_pressure_conversion_and_statistics(tmp_path):
    aifs2 = tmp_path / "aifs2.csv"
    graphcast = tmp_path / "graphcast.csv"
    pangu = tmp_path / "pangu.csv"

    write_track(
        aifs2,
        leads=[0],
        pressures=[97000],
        winds=[35],
    )

    write_track(
        graphcast,
        leads=[0],
        pressures=[96000],
        winds=[40],
    )

    write_track(
        pangu,
        leads=[0],
        pressures=[96500],
        winds=[38],
    )

    result = build_intensity_guidance(
        aifs2_path=aifs2,
        graphcast_path=graphcast,
        pangu_path=pangu,
        storm="Polo",
    )

    row = result.iloc[0]

    assert row["aifs2_pressure_hpa"] == pytest.approx(970)
    assert row["graphcast_pressure_hpa"] == pytest.approx(960)
    assert row["pangu_pressure_hpa"] == pytest.approx(965)

    assert row["pressure_mean_hpa"] == pytest.approx(965)
    assert row["pressure_range_hpa"] == pytest.approx(10)

    assert row["wind_mean_ms"] == pytest.approx(
        (35 + 40 + 38) / 3
    )

    assert row["wind_range_ms"] == pytest.approx(5)


def test_three_hour_records_are_removed(tmp_path):
    paths = []

    for name in ("aifs2", "graphcast", "pangu"):
        path = tmp_path / f"{name}.csv"

        write_track(
            path,
            leads=[0, 3, 6],
            pressures=[99000, 99100, 99200],
            winds=[30, 29, 28],
        )

        paths.append(path)

    result = build_intensity_guidance(
        aifs2_path=paths[0],
        graphcast_path=paths[1],
        pangu_path=paths[2],
        storm="Test Storm",
    )

    assert list(result["lead_time_hours"]) == [0, 6]


def test_invalid_units_raise(tmp_path):
    paths = []

    for name in ("aifs2", "graphcast", "pangu"):
        path = tmp_path / f"{name}.csv"

        write_track(
            path,
            leads=[0],
            pressures=[99000],
            winds=[30],
        )

        paths.append(path)

    df = pd.read_csv(paths[0])
    df["pressure_units"] = "hPa"
    df.to_csv(paths[0], index=False)

    with pytest.raises(
        ValueError,
        match="expected pressure units Pa",
    ):
        build_intensity_guidance(
            aifs2_path=paths[0],
            graphcast_path=paths[1],
            pangu_path=paths[2],
            storm="Test Storm",
        )
