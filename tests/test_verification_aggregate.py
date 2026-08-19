import pandas as pd
import pytest

from aiweather.verification.aggregate import (
    aggregate_verification_summary,
)


def make_summary():
    return pd.DataFrame(
        [
            {
                "case_id": "case_a",
                "model_name": "graphcast",
                "tracker": "native",
                "coverage": "full",
                "overlap_count": 10,
                "mean_track_error_km": 100.0,
                "rmse_track_error_km": 120.0,
                "pressure_mae_pa": 1000.0,
                "wind_mae_ms": 10.0,
                "valid_for_aggregation": True,
            },
            {
                "case_id": "case_b",
                "model_name": "graphcast",
                "tracker": "native",
                "coverage": "full",
                "overlap_count": 20,
                "mean_track_error_km": 300.0,
                "rmse_track_error_km": 320.0,
                "pressure_mae_pa": 2000.0,
                "wind_mae_ms": 20.0,
                "valid_for_aggregation": True,
            },
            {
                "case_id": "case_c",
                "model_name": "graphcast",
                "tracker": "native",
                "coverage": "full",
                "overlap_count": 15,
                "mean_track_error_km": 10000.0,
                "rmse_track_error_km": 10050.0,
                "pressure_mae_pa": 500.0,
                "wind_mae_ms": 5.0,
                "valid_for_aggregation": False,
            },
            {
                "case_id": "case_a",
                "model_name": "graphcast",
                "tracker": "wuduan",
                "coverage": "full",
                "overlap_count": 12,
                "mean_track_error_km": 80.0,
                "rmse_track_error_km": 90.0,
                "pressure_mae_pa": 900.0,
                "wind_mae_ms": 9.0,
                "valid_for_aggregation": True,
            },
            {
                "case_id": "case_b",
                "model_name": "graphcast",
                "tracker": "wuduan",
                "coverage": "full",
                "overlap_count": 18,
                "mean_track_error_km": 200.0,
                "rmse_track_error_km": 220.0,
                "pressure_mae_pa": 1500.0,
                "wind_mae_ms": 15.0,
                "valid_for_aggregation": True,
            },
            {
                "case_id": "case_a",
                "model_name": "graphcast",
                "tracker": "native",
                "coverage": "common",
                "overlap_count": 8,
                "mean_track_error_km": 110.0,
                "rmse_track_error_km": 130.0,
                "pressure_mae_pa": 1100.0,
                "wind_mae_ms": 11.0,
                "valid_for_aggregation": True,
            },
        ]
    )


def test_aggregate_verification_summary():
    summary = make_summary()

    result = aggregate_verification_summary(
        summary
    )

    assert not result.empty

    assert set(
        result["tracker"]
    ) == {
        "native",
        "wuduan",
    }

    assert set(
        result["coverage"]
    ) == {
        "full",
        "common",
    }


def test_aggregate_filters_invalid_rows():
    summary = make_summary()

    result = aggregate_verification_summary(
        summary
    )

    native_full = result[
        (result["tracker"] == "native")
        & (result["coverage"] == "full")
    ].iloc[0]

    assert native_full[
        "case_count"
    ] == 2

    assert native_full[
        "total_overlap_points"
    ] == 30

    assert native_full[
        "mean_case_track_error_km"
    ] == pytest.approx(
        200.0
    )


def test_aggregate_case_level_equal_weighting():
    summary = make_summary()

    result = aggregate_verification_summary(
        summary
    )

    native_full = result[
        (result["tracker"] == "native")
        & (result["coverage"] == "full")
    ].iloc[0]

    # Equal case weighting:
    # (100 + 300) / 2 = 200
    #
    # A pooled-by-overlap calculation would differ,
    # which is intentionally not used here.
    assert native_full[
        "mean_case_track_error_km"
    ] == pytest.approx(
        200.0
    )

    assert native_full[
        "median_case_track_error_km"
    ] == pytest.approx(
        200.0
    )

    assert native_full[
        "mean_case_rmse_track_error_km"
    ] == pytest.approx(
        220.0
    )

    assert native_full[
        "mean_pressure_mae_pa"
    ] == pytest.approx(
        1500.0
    )

    assert native_full[
        "mean_wind_mae_ms"
    ] == pytest.approx(
        15.0
    )


def test_aggregate_separates_trackers():
    summary = make_summary()

    result = aggregate_verification_summary(
        summary
    )

    wuduan_full = result[
        (result["tracker"] == "wuduan")
        & (result["coverage"] == "full")
    ].iloc[0]

    assert wuduan_full[
        "case_count"
    ] == 2

    assert wuduan_full[
        "mean_case_track_error_km"
    ] == pytest.approx(
        140.0
    )

    assert wuduan_full[
        "mean_case_rmse_track_error_km"
    ] == pytest.approx(
        155.0
    )


def test_aggregate_separates_full_and_common():
    summary = make_summary()

    result = aggregate_verification_summary(
        summary
    )

    native_common = result[
        (result["tracker"] == "native")
        & (result["coverage"] == "common")
    ].iloc[0]

    assert native_common[
        "case_count"
    ] == 1

    assert native_common[
        "total_overlap_points"
    ] == 8

    assert native_common[
        "mean_case_track_error_km"
    ] == pytest.approx(
        110.0
    )


def test_aggregate_separates_models():
    summary = make_summary()

    second_model = summary.iloc[
        [0]
    ].copy()

    second_model[
        "case_id"
    ] = "case_other"

    second_model[
        "model_name"
    ] = "pangu"

    second_model[
        "mean_track_error_km"
    ] = 50.0

    combined = pd.concat(
        [
            summary,
            second_model,
        ],
        ignore_index=True,
    )

    result = aggregate_verification_summary(
        combined
    )

    assert set(
        result["model_name"]
    ) == {
        "graphcast",
        "pangu",
    }


def test_aggregate_counts_unique_cases():
    summary = make_summary()

    duplicate = summary.iloc[
        [0]
    ].copy()

    combined = pd.concat(
        [
            summary,
            duplicate,
        ],
        ignore_index=True,
    )

    result = aggregate_verification_summary(
        combined
    )

    native_full = result[
        (result["tracker"] == "native")
        & (result["coverage"] == "full")
    ].iloc[0]

    assert native_full[
        "case_count"
    ] == 2


def test_aggregate_empty_valid_set():
    summary = make_summary()

    summary[
        "valid_for_aggregation"
    ] = False

    result = aggregate_verification_summary(
        summary
    )

    assert result.empty

    assert list(
        result.columns
    ) == [
        "model_name",
        "tracker",
        "coverage",
        "case_count",
        "total_overlap_points",
        "mean_case_track_error_km",
        "median_case_track_error_km",
        "mean_case_rmse_track_error_km",
        "mean_pressure_mae_pa",
        "mean_wind_mae_ms",
    ]


def test_aggregate_invalid_input_type():
    with pytest.raises(
        TypeError,
        match="pandas DataFrame",
    ):
        aggregate_verification_summary(
            []
        )


def test_aggregate_missing_required_columns():
    summary = pd.DataFrame(
        {
            "case_id": [
                "case_a",
            ],
            "model_name": [
                "graphcast",
            ],
        }
    )

    with pytest.raises(
        ValueError,
        match="missing required columns",
    ):
        aggregate_verification_summary(
            summary
        )


def test_aggregate_preserves_column_order():
    result = aggregate_verification_summary(
        make_summary()
    )

    assert list(
        result.columns
    ) == [
        "model_name",
        "tracker",
        "coverage",
        "case_count",
        "total_overlap_points",
        "mean_case_track_error_km",
        "median_case_track_error_km",
        "mean_case_rmse_track_error_km",
        "mean_pressure_mae_pa",
        "mean_wind_mae_ms",
    ]