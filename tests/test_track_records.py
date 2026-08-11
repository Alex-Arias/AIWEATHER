from datetime import datetime

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from aiweather.tracking import (
    TrackPoint,
    TrackRecord,
    build_track_records,
    records_to_dataframe,
    records_to_xarray,
)


def make_track():
    return [
        TrackPoint(
            lead_time_hours=0,
            latitude=10.0,
            longitude=250.0,
            pressure=100000.0,
            pressure_units="Pa",
            max_wind=10.0,
            wind_units="m s-1",
        ),
        TrackPoint(
            lead_time_hours=6,
            latitude=11.0,
            longitude=250.0,
            pressure=99500.0,
            pressure_units="Pa",
            max_wind=12.0,
            wind_units="m s-1",
        ),
        TrackPoint(
            lead_time_hours=12,
            latitude=12.0,
            longitude=250.0,
            pressure=99000.0,
            pressure_units="Pa",
            max_wind=15.0,
            wind_units="m s-1",
        ),
    ]


# ---------------------------------------------------------
# TrackRecord
# ---------------------------------------------------------


def test_build_track_records_length():
    records = build_track_records(
        make_track(),
        initialization_time="2026-07-24T00:00:00",
    )

    assert len(records) == 3

    assert all(
        isinstance(record, TrackRecord)
        for record in records
    )


def test_build_track_records_valid_times():
    records = build_track_records(
        make_track(),
        initialization_time="2026-07-24T00:00:00",
    )

    assert records[0].valid_time == np.datetime64(
        "2026-07-24T00:00:00"
    )

    assert records[1].valid_time == np.datetime64(
        "2026-07-24T06:00:00"
    )

    assert records[2].valid_time == np.datetime64(
        "2026-07-24T12:00:00"
    )


def test_build_track_records_accepts_datetime():
    records = build_track_records(
        make_track(),
        initialization_time=datetime(
            2026,
            7,
            24,
            0,
            0,
        ),
    )

    assert records[1].valid_time == np.datetime64(
        "2026-07-24T06:00:00"
    )


def test_first_record_has_no_motion():
    records = build_track_records(
        make_track(),
        initialization_time="2026-07-24T00:00:00",
    )

    first = records[0]

    assert first.distance_km is None
    assert first.translation_speed_kmh is None
    assert first.bearing_degrees is None
    assert first.cumulative_distance_km == pytest.approx(
        0.0
    )


def test_later_records_have_motion():
    records = build_track_records(
        make_track(),
        initialization_time="2026-07-24T00:00:00",
    )

    second = records[1]

    assert second.distance_km is not None
    assert second.translation_speed_kmh is not None
    assert second.bearing_degrees is not None

    assert second.cumulative_distance_km > 0.0


def test_build_track_records_empty_track():
    records = build_track_records(
        [],
        initialization_time="2026-07-24T00:00:00",
    )

    assert records == []


def test_build_track_records_rejects_invalid_track():
    with pytest.raises(TypeError):
        build_track_records(
            ["not a TrackPoint"],
            initialization_time="2026-07-24T00:00:00",
        )


def test_build_track_records_rejects_nat():
    with pytest.raises(ValueError):
        build_track_records(
            make_track(),
            initialization_time=np.datetime64("NaT"),
        )


# ---------------------------------------------------------
# DataFrame conversion
# ---------------------------------------------------------


def test_records_to_dataframe():
    records = build_track_records(
        make_track(),
        initialization_time="2026-07-24T00:00:00",
    )

    dataframe = records_to_dataframe(
        records
    )

    assert isinstance(
        dataframe,
        pd.DataFrame,
    )

    assert len(dataframe) == 3

    assert list(dataframe["lead_time_hours"]) == [
        0,
        6,
        12,
    ]

    assert pd.api.types.is_datetime64_any_dtype(
        dataframe["valid_time"]
    )


def test_records_to_dataframe_preserves_values():
    records = build_track_records(
        make_track(),
        initialization_time="2026-07-24T00:00:00",
    )

    dataframe = records_to_dataframe(
        records
    )

    assert dataframe.loc[1, "latitude"] == pytest.approx(
        11.0
    )

    assert dataframe.loc[1, "pressure"] == pytest.approx(
        99500.0
    )

    assert dataframe.loc[1, "max_wind"] == pytest.approx(
        12.0
    )


def test_records_to_dataframe_empty():
    dataframe = records_to_dataframe(
        []
    )

    assert isinstance(
        dataframe,
        pd.DataFrame,
    )

    assert dataframe.empty


# ---------------------------------------------------------
# xarray conversion
# ---------------------------------------------------------


def test_records_to_xarray():
    records = build_track_records(
        make_track(),
        initialization_time="2026-07-24T00:00:00",
    )

    dataset = records_to_xarray(
        records
    )

    assert isinstance(
        dataset,
        xr.Dataset,
    )

    assert dataset.sizes["track_point"] == 3

    assert "valid_time" in dataset.coords
    assert "latitude" in dataset
    assert "longitude" in dataset
    assert "pressure" in dataset
    assert "max_wind" in dataset
    assert "translation_speed_kmh" in dataset


def test_records_to_xarray_valid_time():
    records = build_track_records(
        make_track(),
        initialization_time="2026-07-24T00:00:00",
    )

    dataset = records_to_xarray(
        records
    )

    assert dataset["valid_time"].values[2] == (
        np.datetime64(
            "2026-07-24T12:00:00.000000000"
        )
    )


def test_records_to_xarray_preserves_units():
    records = build_track_records(
        make_track(),
        initialization_time="2026-07-24T00:00:00",
    )

    dataset = records_to_xarray(
        records
    )

    assert dataset["pressure"].attrs["units"] == "Pa"
    assert dataset["max_wind"].attrs["units"] == "m s-1"

    assert dataset["distance_km"].attrs["units"] == "km"

    assert (
        dataset["translation_speed_kmh"].attrs["units"]
        == "km h-1"
    )


def test_records_to_xarray_first_motion_is_nan():
    records = build_track_records(
        make_track(),
        initialization_time="2026-07-24T00:00:00",
    )

    dataset = records_to_xarray(
        records
    )

    assert np.isnan(
        dataset["distance_km"].values[0]
    )

    assert np.isnan(
        dataset["translation_speed_kmh"].values[0]
    )

    assert np.isnan(
        dataset["bearing_degrees"].values[0]
    )


def test_records_to_xarray_empty():
    dataset = records_to_xarray(
        []
    )

    assert isinstance(
        dataset,
        xr.Dataset,
    )

    assert dataset.sizes["track_point"] == 0
