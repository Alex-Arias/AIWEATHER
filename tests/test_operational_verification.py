import pytest
import numpy as np
import pandas as pd

from aiweather.tracking import TrackRecord

from aiweather.verification.operational import (
    validate_operational_track_units,
    verify_operational_track,
    verify_operational_track_csv,
)

def make_record(
    *,
    pressure_units=None,
    wind_units=None,
):
    return TrackRecord(
        lead_time_hours=0,
        valid_time="2026-09-05T00:00:00",
        latitude=20.5,
        longitude=241.25,
        pressure=96724.0,
        pressure_units=pressure_units,
        max_wind=26.8,
        wind_units=wind_units,
    )


def test_validate_operational_track_units_legacy_blank():
    validate_operational_track_units(
        [make_record()]
    )


def test_validate_operational_track_units_canonical():
    validate_operational_track_units(
        [
            make_record(
                pressure_units="Pa",
                wind_units="m/s",
            )
        ]
    )


def test_validate_operational_track_units_rejects_pressure():
    with pytest.raises(
        ValueError,
        match="pressure units",
    ):
        validate_operational_track_units(
            [
                make_record(
                    pressure_units="hPa",
                )
            ]
        )


def test_validate_operational_track_units_rejects_wind():
    with pytest.raises(
        ValueError,
        match="wind units",
    ):
        validate_operational_track_units(
            [
                make_record(
                    wind_units="knots",
                )
            ]
        )


def test_validate_operational_track_units_rejects_non_list():
    with pytest.raises(
        TypeError,
        match="list",
    ):
        validate_operational_track_units(
            ()
        )


def test_validate_operational_track_units_rejects_contents():
    with pytest.raises(
        TypeError,
        match="TrackRecord",
    ):
        validate_operational_track_units(
            ["not-a-record"]
        )


def test_verify_operational_track():
    forecast = [
        TrackRecord(
            lead_time_hours=0,
            valid_time=np.datetime64(
                "2026-09-05T00:00:00"
            ),
            latitude=20.5,
            longitude=241.25,
            pressure=97000.0,
            max_wind=25.0,
        ),
        TrackRecord(
            lead_time_hours=6,
            valid_time=np.datetime64(
                "2026-09-05T06:00:00"
            ),
            latitude=21.0,
            longitude=240.5,
            pressure=96500.0,
            max_wind=27.0,
        ),
    ]

    observations = [
        TrackRecord(
            lead_time_hours=0,
            valid_time=np.datetime64(
                "2026-09-05T00:00:00"
            ),
            latitude=20.0,
            longitude=241.0,
            pressure=96800.0,
            pressure_units="Pa",
            max_wind=26.0,
            wind_units="m/s",
        ),
        TrackRecord(
            lead_time_hours=6,
            valid_time=np.datetime64(
                "2026-09-05T06:00:00"
            ),
            latitude=20.5,
            longitude=240.0,
            pressure=96000.0,
            pressure_units="Pa",
            max_wind=29.0,
            wind_units="m/s",
        ),
    ]

    result = verify_operational_track(
        forecast,
        observations,
        forecast_name="AIFS2",
        observation_name="IBTrACS",
    )

    assert result.forecast_name == "AIFS2"
    assert result.observation_name == "IBTrACS"
    assert result.overlap_count == 2
    assert result.table[
        "pressure_error_pa"
    ].tolist() == pytest.approx(
        [200.0, 500.0]
    )
    assert result.table[
        "wind_error_ms"
    ].tolist() == pytest.approx(
        [-1.0, -2.0]
    )


def test_verify_operational_track_rejects_bad_units():
    forecast = [
        make_record(
            pressure_units="hPa",
        )
    ]

    observations = [
        make_record(
            pressure_units="Pa",
            wind_units="m/s",
        )
    ]

    with pytest.raises(
        ValueError,
        match="pressure units",
    ):
        verify_operational_track(
            forecast,
            observations,
        )


def test_verify_operational_track_csv(
    tmp_path,
):
    forecast_path = (
        tmp_path / "forecast.csv"
    )
    ibtracs_path = (
        tmp_path / "ibtracs.csv"
    )

    pd.DataFrame(
        {
            "lead_time_hours": [0, 6],
            "valid_time": [
                "2026-09-05 00:00:00",
                "2026-09-05 06:00:00",
            ],
            "latitude": [20.5, 21.0],
            "longitude": [241.25, 240.5],
            "pressure": [
                97000.0,
                96500.0,
            ],
            "pressure_units": [
                None,
                None,
            ],
            "max_wind": [
                25.0,
                27.0,
            ],
            "wind_units": [
                None,
                None,
            ],
        }
    ).to_csv(
        forecast_path,
        index=False,
    )

    pd.DataFrame(
        {
            "SID": [
                "2026248N20241",
                "2026248N20241",
            ],
            "ISO_TIME": [
                "2026-09-05 00:00:00",
                "2026-09-05 06:00:00",
            ],
            "LAT": [
                20.0,
                20.5,
            ],
            "LON": [
                -119.0,
                -120.0,
            ],
            "USA_WIND": [
                50.0,
                55.0,
            ],
            "USA_PRES": [
                968.0,
                960.0,
            ],
        }
    ).to_csv(
        ibtracs_path,
        index=False,
    )

    result = verify_operational_track_csv(
        forecast_path,
        ibtracs_path,
        sid="2026248N20241",
        initialization_time=(
            "2026-09-05T00:00:00"
        ),
        forecast_name="AIFS2",
    )

    assert result.forecast_name == "AIFS2"
    assert result.observation_name == "IBTrACS"
    assert result.overlap_count == 2

    assert result.table[
        "pressure_error_pa"
    ].tolist() == pytest.approx(
        [200.0, 500.0]
    )