import pandas as pd
import pytest

from aiweather.verification.comparison import (
    TrackVerification,
)
from aiweather.verification.qc import (
    TrackerQCResult,
    evaluate_tracker_qc,
)


def make_verification(
    errors,
):
    table = pd.DataFrame(
        {
            "valid_time": pd.date_range(
                "2026-07-24T00:00:00",
                periods=len(errors),
                freq="6h",
            ),
            "track_error_km": errors,
        }
    )

    return TrackVerification(
        forecast_name="test",
        observation_name="ibtracs_test",
        table=table,
    )


def test_tracker_qc_pass():
    verification = make_verification(
        [
            120.0,
            150.0,
            180.0,
            220.0,
            250.0,
            300.0,
        ]
    )

    result = evaluate_tracker_qc(
        verification
    )

    assert isinstance(
        result,
        TrackerQCResult,
    )

    assert result.status == "pass"
    assert result.reason is None
    assert result.overlap_count == 6

    assert result.initial_separation_km == pytest.approx(
        120.0
    )

    assert (
        result.valid_for_aggregation
        is True
    )


def test_tracker_qc_limited_short_overlap():
    verification = make_verification(
        [
            100.0,
            120.0,
            140.0,
            160.0,
            180.0,
        ]
    )

    result = evaluate_tracker_qc(
        verification
    )

    assert result.status == "limited"
    assert result.reason == "short_overlap"
    assert result.overlap_count == 5

    assert result.initial_separation_km == pytest.approx(
        100.0
    )

    assert (
        result.valid_for_aggregation
        is False
    )


def test_tracker_qc_fail_initial_association():
    verification = make_verification(
        [
            10500.0,
            10600.0,
            10700.0,
            10800.0,
            10900.0,
            11000.0,
        ]
    )

    result = evaluate_tracker_qc(
        verification
    )

    assert result.status == "fail"

    assert (
        result.reason
        == "initial_association"
    )

    assert result.overlap_count == 6

    assert result.initial_separation_km == pytest.approx(
        10500.0
    )

    assert (
        result.valid_for_aggregation
        is False
    )


def test_tracker_qc_fail_zero_overlap():
    verification = TrackVerification(
        forecast_name="test",
        observation_name="ibtracs_test",
        table=pd.DataFrame(
            columns=[
                "valid_time",
                "track_error_km",
            ]
        ),
    )

    result = evaluate_tracker_qc(
        verification
    )

    assert result.status == "fail"
    assert result.reason == "no_overlap"
    assert result.overlap_count == 0

    assert (
        result.initial_separation_km
        is None
    )

    assert (
        result.valid_for_aggregation
        is False
    )


def test_tracker_qc_custom_initial_separation_threshold():
    verification = make_verification(
        [
            1500.0,
            1600.0,
            1700.0,
            1800.0,
            1900.0,
            1950.0,
        ]
    )

    result = evaluate_tracker_qc(
        verification,
        maximum_initial_separation_km=1000.0,
    )

    assert result.status == "fail"

    assert (
        result.reason
        == "initial_association"
    )


def test_tracker_qc_custom_minimum_overlap():
    verification = make_verification(
        [
            100.0,
            120.0,
            140.0,
            160.0,
            180.0,
            200.0,
        ]
    )

    result = evaluate_tracker_qc(
        verification,
        minimum_overlap=10,
    )

    assert result.status == "limited"
    assert result.reason == "short_overlap"


def test_tracker_qc_invalid_type():
    with pytest.raises(
        TypeError,
        match="TrackVerification",
    ):
        evaluate_tracker_qc(
            "not-a-verification"
        )


def test_tracker_qc_invalid_initial_threshold():
    verification = make_verification(
        [
            100.0,
        ]
    )

    with pytest.raises(
        ValueError,
        match=(
            "maximum_initial_separation_km "
            "must be positive"
        ),
    ):
        evaluate_tracker_qc(
            verification,
            maximum_initial_separation_km=0.0,
        )


def test_tracker_qc_invalid_minimum_overlap():
    verification = make_verification(
        [
            100.0,
        ]
    )

    with pytest.raises(
        ValueError,
        match="minimum_overlap must be at least 1",
    ):
        evaluate_tracker_qc(
            verification,
            minimum_overlap=0,
        )


def test_tracker_qc_requires_track_error_column():
    verification = TrackVerification(
        forecast_name="test",
        observation_name="ibtracs_test",
        table=pd.DataFrame(
            {
                "valid_time": pd.date_range(
                    "2026-07-24",
                    periods=6,
                    freq="6h",
                ),
            }
        ),
    )

    with pytest.raises(
        ValueError,
        match="track_error_km",
    ):
        evaluate_tracker_qc(
            verification
        )


def test_tracker_qc_invalid_initial_separation_nan():
    verification = make_verification(
        [
            float("nan"),
            100.0,
            120.0,
            140.0,
            160.0,
            180.0,
        ]
    )

    result = evaluate_tracker_qc(
        verification
    )

    assert result.status == "fail"

    assert (
        result.reason
        == "invalid_initial_separation"
    )

    assert (
        result.initial_separation_km
        is None
    )

    assert (
        result.valid_for_aggregation
        is False
    )