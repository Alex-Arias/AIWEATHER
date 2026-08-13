import numpy as np
import pytest

from aiweather.tracking.records import TrackRecord
from aiweather.verification.comparison import (
    TrackVerification,
    align_by_valid_time,
    compare_forecast_to_best_track,
)


def make_record(
    valid_time,
    lead_time_hours,
    latitude,
    longitude,
    *,
    pressure=100000.0,
    max_wind=20.0,
):
    return TrackRecord(
        lead_time_hours=lead_time_hours,
        valid_time=np.datetime64(
            valid_time
        ),
        latitude=latitude,
        longitude=longitude,
        pressure=pressure,
        pressure_units="Pa",
        max_wind=max_wind,
        wind_units="m/s",
    )


def test_align_by_valid_time():
    forecast = [
        make_record(
            "2026-07-24T00:00:00",
            0,
            10.0,
            250.0,
        ),
        make_record(
            "2026-07-24T06:00:00",
            6,
            11.0,
            249.0,
        ),
    ]

    observations = [
        make_record(
            "2026-07-24T00:00:00",
            0,
            10.1,
            250.1,
        ),
        make_record(
            "2026-07-24T03:00:00",
            3,
            10.5,
            249.5,
        ),
        make_record(
            "2026-07-24T06:00:00",
            6,
            11.1,
            249.1,
        ),
    ]

    aligned = align_by_valid_time(
        forecast,
        observations,
    )

    assert len(aligned) == 2

    assert aligned[0][0].lead_time_hours == 0
    assert aligned[1][0].lead_time_hours == 6


def test_compare_forecast_to_best_track_exact():
    forecast = [
        make_record(
            "2026-07-24T00:00:00",
            0,
            10.0,
            250.0,
        ),
    ]

    observations = [
        make_record(
            "2026-07-24T00:00:00",
            0,
            10.0,
            250.0,
        ),
    ]

    result = compare_forecast_to_best_track(
        forecast,
        observations,
    )

    assert isinstance(
        result,
        TrackVerification,
    )

    assert result.overlap_count == 1

    assert result.mean_track_error_km == pytest.approx(
        0.0
    )

    assert result.mean_pressure_error_pa == pytest.approx(
        0.0
    )

    assert result.mean_wind_error_ms == pytest.approx(
        0.0
    )


def test_compare_forecast_to_best_track_intensity_errors():
    forecast = [
        make_record(
            "2026-07-24T00:00:00",
            0,
            10.0,
            250.0,
            pressure=100000.0,
            max_wind=20.0,
        ),
    ]

    observations = [
        make_record(
            "2026-07-24T00:00:00",
            0,
            10.0,
            250.0,
            pressure=98000.0,
            max_wind=40.0,
        ),
    ]

    result = compare_forecast_to_best_track(
        forecast,
        observations,
    )

    assert result.mean_pressure_error_pa == pytest.approx(
        2000.0
    )

    assert result.mean_absolute_pressure_error_pa == pytest.approx(
        2000.0
    )

    assert result.rmse_pressure_error_pa == pytest.approx(
        2000.0
    )

    assert result.mean_wind_error_ms == pytest.approx(
        -20.0
    )

    assert result.mean_absolute_wind_error_ms == pytest.approx(
        20.0
    )

    assert result.rmse_wind_error_ms == pytest.approx(
        20.0
    )


def test_compare_forecast_to_best_track_missing_intensity():
    forecast = [
        make_record(
            "2026-07-24T00:00:00",
            0,
            10.0,
            250.0,
            pressure=None,
            max_wind=None,
        ),
    ]

    observations = [
        make_record(
            "2026-07-24T00:00:00",
            0,
            10.0,
            250.0,
        ),
    ]

    result = compare_forecast_to_best_track(
        forecast,
        observations,
    )

    assert np.isnan(
        result.mean_pressure_error_pa
    )

    assert np.isnan(
        result.mean_wind_error_ms
    )


def test_compare_forecast_to_best_track_no_overlap():
    forecast = [
        make_record(
            "2026-07-24T00:00:00",
            0,
            10.0,
            250.0,
        ),
    ]

    observations = [
        make_record(
            "2026-07-24T03:00:00",
            3,
            10.0,
            250.0,
        ),
    ]

    result = compare_forecast_to_best_track(
        forecast,
        observations,
    )

    assert result.overlap_count == 0
    assert np.isnan(
        result.mean_track_error_km
    )


def test_align_by_valid_time_invalid_forecast():
    with pytest.raises(
        TypeError,
        match="forecast_records",
    ):
        align_by_valid_time(
            "invalid",
            [],
        )


def test_align_by_valid_time_invalid_observations():
    with pytest.raises(
        TypeError,
        match="observation_records",
    ):
        align_by_valid_time(
            [],
            "invalid",
        )

def test_common_overlap_verifications():
    from aiweather.verification.comparison import (
        common_overlap_verifications,
    )

    native = compare_forecast_to_best_track(
        [
            make_record(
                "2026-07-24T00:00:00",
                0,
                10.0,
                250.0,
            ),
            make_record(
                "2026-07-24T06:00:00",
                6,
                11.0,
                249.0,
            ),
            make_record(
                "2026-07-24T12:00:00",
                12,
                12.0,
                248.0,
            ),
        ],
        [
            make_record(
                "2026-07-24T00:00:00",
                0,
                10.0,
                250.0,
            ),
            make_record(
                "2026-07-24T06:00:00",
                6,
                11.0,
                249.0,
            ),
            make_record(
                "2026-07-24T12:00:00",
                12,
                12.0,
                248.0,
            ),
        ],
        forecast_name="native",
    )

    wuduan = compare_forecast_to_best_track(
        [
            make_record(
                "2026-07-24T06:00:00",
                6,
                11.0,
                249.0,
            ),
            make_record(
                "2026-07-24T12:00:00",
                12,
                12.0,
                248.0,
            ),
            make_record(
                "2026-07-24T18:00:00",
                18,
                13.0,
                247.0,
            ),
        ],
        [
            make_record(
                "2026-07-24T06:00:00",
                6,
                11.0,
                249.0,
            ),
            make_record(
                "2026-07-24T12:00:00",
                12,
                12.0,
                248.0,
            ),
            make_record(
                "2026-07-24T18:00:00",
                18,
                13.0,
                247.0,
            ),
        ],
        forecast_name="wuduan",
    )

    vitart = compare_forecast_to_best_track(
        [
            make_record(
                "2026-07-24T12:00:00",
                12,
                12.0,
                248.0,
            ),
            make_record(
                "2026-07-24T18:00:00",
                18,
                13.0,
                247.0,
            ),
        ],
        [
            make_record(
                "2026-07-24T12:00:00",
                12,
                12.0,
                248.0,
            ),
            make_record(
                "2026-07-24T18:00:00",
                18,
                13.0,
                247.0,
            ),
        ],
        forecast_name="vitart",
    )

    result = common_overlap_verifications(
        {
            "native": native,
            "wuduan": wuduan,
            "vitart": vitart,
        }
    )

    assert set(result) == {
        "native",
        "wuduan",
        "vitart",
    }

    for verification in result.values():
        assert verification.overlap_count == 1

        assert list(
            verification.table[
                "lead_time_hours"
            ]
        ) == [12]


def test_common_overlap_verifications_empty():
    from aiweather.verification.comparison import (
        common_overlap_verifications,
    )

    result = common_overlap_verifications(
        {}
    )

    assert result == {}


def test_common_overlap_verifications_invalid_type():
    from aiweather.verification.comparison import (
        common_overlap_verifications,
    )

    with pytest.raises(
        TypeError,
        match="dictionary",
    ):
        common_overlap_verifications(
            []
        )


def test_common_overlap_verifications_invalid_contents():
    from aiweather.verification.comparison import (
        common_overlap_verifications,
    )

    with pytest.raises(
        TypeError,
        match="TrackVerification",
    ):
        common_overlap_verifications(
            {
                "native": object(),
            }
        )