import numpy as np
import pytest

from aiweather.tracking import (
    TrackRecord,
    classify_track,
)


def make_record(
    lead_time_hours,
    *,
    pressure=100000.0,
    max_wind=15.0,
):
    return TrackRecord(
        lead_time_hours=lead_time_hours,
        valid_time=(
            np.datetime64("2026-07-24T00:00:00")
            + np.timedelta64(
                lead_time_hours,
                "h",
            )
        ),
        latitude=15.0,
        longitude=250.0,
        pressure=pressure,
        max_wind=max_wind,
    )


# ---------------------------------------------------------
# Basic criteria
# ---------------------------------------------------------


def test_classify_track_pressure_and_wind():
    records = [
        make_record(
            0,
            pressure=100000.0,
            max_wind=15.0,
        ),
    ]

    result = classify_track(
        records,
        minimum_wind=10.0,
        maximum_pressure=101000.0,
        minimum_consecutive_points=1,
    )

    classification = result[0]

    assert classification.wind_threshold_met
    assert classification.pressure_threshold_met
    assert classification.intensity_criteria_met
    assert classification.persistence_met
    assert classification.is_candidate


def test_classify_track_wind_failure():
    records = [
        make_record(
            0,
            pressure=99000.0,
            max_wind=5.0,
        ),
    ]

    result = classify_track(
        records,
        minimum_wind=10.0,
        maximum_pressure=101000.0,
        minimum_consecutive_points=1,
    )

    assert result[0].pressure_threshold_met
    assert not result[0].wind_threshold_met
    assert not result[0].is_candidate


def test_classify_track_pressure_failure():
    records = [
        make_record(
            0,
            pressure=102000.0,
            max_wind=20.0,
        ),
    ]

    result = classify_track(
        records,
        minimum_wind=10.0,
        maximum_pressure=101000.0,
        minimum_consecutive_points=1,
    )

    assert result[0].wind_threshold_met
    assert not result[0].pressure_threshold_met
    assert not result[0].is_candidate


# ---------------------------------------------------------
# Persistence
# ---------------------------------------------------------


def test_classification_requires_persistence():
    records = [
        make_record(
            0,
            pressure=100000.0,
            max_wind=15.0,
        ),
        make_record(
            6,
            pressure=100000.0,
            max_wind=15.0,
        ),
        make_record(
            12,
            pressure=102000.0,
            max_wind=5.0,
        ),
    ]

    result = classify_track(
        records,
        minimum_wind=10.0,
        maximum_pressure=101000.0,
        minimum_consecutive_points=3,
    )

    assert result[0].intensity_criteria_met
    assert result[1].intensity_criteria_met

    assert not result[0].persistence_met
    assert not result[1].persistence_met

    assert not any(
        item.is_candidate
        for item in result
    )


def test_persistent_run_marks_complete_run():
    records = [
        make_record(
            0,
            pressure=102000.0,
            max_wind=5.0,
        ),
        make_record(
            6,
            pressure=100000.0,
            max_wind=15.0,
        ),
        make_record(
            12,
            pressure=99500.0,
            max_wind=16.0,
        ),
        make_record(
            18,
            pressure=99000.0,
            max_wind=17.0,
        ),
        make_record(
            24,
            pressure=102000.0,
            max_wind=5.0,
        ),
    ]

    result = classify_track(
        records,
        minimum_wind=10.0,
        maximum_pressure=101000.0,
        minimum_consecutive_points=3,
    )

    assert not result[0].is_candidate

    assert result[1].is_candidate
    assert result[2].is_candidate
    assert result[3].is_candidate

    assert not result[4].is_candidate


# ---------------------------------------------------------
# Optional criteria
# ---------------------------------------------------------


def test_pressure_only_classification():
    records = [
        make_record(
            0,
            pressure=100000.0,
            max_wind=None,
        ),
    ]

    result = classify_track(
        records,
        maximum_pressure=101000.0,
        minimum_consecutive_points=1,
    )

    assert result[0].pressure_threshold_met
    assert result[0].is_candidate


def test_wind_only_classification():
    records = [
        make_record(
            0,
            pressure=105000.0,
            max_wind=15.0,
        ),
    ]

    result = classify_track(
        records,
        minimum_wind=10.0,
        minimum_consecutive_points=1,
    )

    assert result[0].wind_threshold_met
    assert result[0].is_candidate


def test_any_enabled_criterion_mode():
    records = [
        make_record(
            0,
            pressure=102000.0,
            max_wind=15.0,
        ),
    ]

    result = classify_track(
        records,
        minimum_wind=10.0,
        maximum_pressure=101000.0,
        minimum_consecutive_points=1,
        require_all_enabled_criteria=False,
    )

    assert result[0].wind_threshold_met
    assert not result[0].pressure_threshold_met
    assert result[0].is_candidate


# ---------------------------------------------------------
# Validation
# ---------------------------------------------------------


def test_classify_track_rejects_no_criteria():
    records = [
        make_record(0),
    ]

    with pytest.raises(ValueError):
        classify_track(
            records
        )


def test_classify_track_rejects_invalid_persistence():
    records = [
        make_record(0),
    ]

    with pytest.raises(ValueError):
        classify_track(
            records,
            minimum_wind=10.0,
            minimum_consecutive_points=0,
        )


def test_classify_track_rejects_negative_wind_threshold():
    records = [
        make_record(0),
    ]

    with pytest.raises(ValueError):
        classify_track(
            records,
            minimum_wind=-1.0,
        )


def test_classify_track_empty_records():
    result = classify_track(
        [],
        minimum_wind=10.0,
    )

    assert result == []


def test_classify_track_rejects_non_record():
    with pytest.raises(TypeError):
        classify_track(
            ["not a TrackRecord"],
            minimum_wind=10.0,
        )


def test_classify_track_rejects_non_increasing_lead_times():
    records = [
        make_record(6),
        make_record(6),
    ]

    with pytest.raises(ValueError):
        classify_track(
            records,
            minimum_wind=10.0,
        )
