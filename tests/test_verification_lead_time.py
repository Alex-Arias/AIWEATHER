import numpy as np
import pandas as pd
import pytest

from aiweather.verification.lead_time import (
    DEFAULT_LEAD_TIME_BINS,
    LeadTimeBin,
    aggregate_batch_lead_time_verification,
    assign_lead_time_bins,
    summarize_lead_time_verification,
)

def make_table():
    return pd.DataFrame(
        {
            "lead_time_hours": [
                0.0,
                6.0,
                18.0,
                24.0,
                30.0,
                48.0,
                54.0,
                72.0,
                96.0,
                120.0,
                144.0,
                168.0,
                192.0,
                240.0,
            ],
            "track_error_km": [
                10.0,
                20.0,
                30.0,
                40.0,
                50.0,
                60.0,
                70.0,
                80.0,
                100.0,
                120.0,
                140.0,
                160.0,
                180.0,
                200.0,
            ],
            "pressure_error_pa": [
                100.0,
                -200.0,
                300.0,
                -400.0,
                500.0,
                -600.0,
                700.0,
                -800.0,
                1000.0,
                -1200.0,
                1400.0,
                -1600.0,
                1800.0,
                -2000.0,
            ],
            "wind_error_ms": [
                1.0,
                -2.0,
                3.0,
                -4.0,
                5.0,
                -6.0,
                7.0,
                -8.0,
                10.0,
                -12.0,
                14.0,
                -16.0,
                18.0,
                -20.0,
            ],
        }
    )


def test_default_lead_time_bins():
    assert DEFAULT_LEAD_TIME_BINS == (
        (0.0, 24.0),
        (24.0, 48.0),
        (48.0, 72.0),
        (72.0, 120.0),
        (120.0, 168.0),
        (168.0, 240.0),
    )


def test_lead_time_bin_dataclass():
    interval = LeadTimeBin(
        start_hours=24.0,
        end_hours=48.0,
    )

    assert interval.start_hours == 24.0
    assert interval.end_hours == 48.0


def test_assign_lead_time_bins_boundaries():
    lead_times = [
        0.0,
        23.999,
        24.0,
        47.999,
        48.0,
        71.999,
        72.0,
        119.999,
        120.0,
        167.999,
        168.0,
        239.999,
        240.0,
    ]

    result = assign_lead_time_bins(
        lead_times
    )

    assert list(
        result.astype(object)
    ) == [
        "0-24h",
        "0-24h",
        "24-48h",
        "24-48h",
        "48-72h",
        "48-72h",
        "72-120h",
        "72-120h",
        "120-168h",
        "120-168h",
        "168-240h",
        "168-240h",
        "168-240h",
    ]


def test_assign_lead_time_bins_outside_range():
    result = assign_lead_time_bins(
        [
            -6.0,
            241.0,
            np.nan,
        ]
    )

    values = list(
        result.astype(object)
    )

    assert pd.isna(
        values[0]
    )

    assert pd.isna(
        values[1]
    )

    assert pd.isna(
        values[2]
    )


def test_assign_lead_time_bins_custom_bins():
    result = assign_lead_time_bins(
        [
            0.0,
            12.0,
            24.0,
            36.0,
            48.0,
        ],
        bins=(
            (0.0, 24.0),
            (24.0, 48.0),
        ),
    )

    assert list(
        result.astype(object)
    ) == [
        "0-24h",
        "0-24h",
        "24-48h",
        "24-48h",
        "24-48h",
    ]


def test_summarize_lead_time_verification():
    result = summarize_lead_time_verification(
        make_table()
    )

    assert list(
        result["lead_time_bin"]
    ) == [
        "0-24h",
        "24-48h",
        "48-72h",
        "72-120h",
        "120-168h",
        "168-240h",
    ]

    assert list(
        result["point_count"]
    ) == [
        3,
        2,
        2,
        2,
        2,
        3,
    ]


def test_summarize_first_bin_metrics():
    result = summarize_lead_time_verification(
        make_table()
    )

    first = result[
        result["lead_time_bin"]
        == "0-24h"
    ].iloc[0]

    assert first[
        "lead_time_start_hours"
    ] == pytest.approx(
        0.0
    )

    assert first[
        "lead_time_end_hours"
    ] == pytest.approx(
        24.0
    )

    assert first[
        "point_count"
    ] == 3

    assert first[
        "mean_track_error_km"
    ] == pytest.approx(
        20.0
    )

    assert first[
        "median_track_error_km"
    ] == pytest.approx(
        20.0
    )

    assert first[
        "rmse_track_error_km"
    ] == pytest.approx(
        np.sqrt(
            (
                10.0 ** 2
                + 20.0 ** 2
                + 30.0 ** 2
            )
            / 3.0
        )
    )

    assert first[
        "pressure_mae_pa"
    ] == pytest.approx(
        200.0
    )

    assert first[
        "wind_mae_ms"
    ] == pytest.approx(
        2.0
    )


def test_summarize_final_bin_includes_240_hours():
    result = summarize_lead_time_verification(
        make_table()
    )

    final = result[
        result["lead_time_bin"]
        == "168-240h"
    ].iloc[0]

    assert final[
        "point_count"
    ] == 3

    assert final[
        "mean_track_error_km"
    ] == pytest.approx(
        (
            160.0
            + 180.0
            + 200.0
        )
        / 3.0
    )


def test_summarize_drops_values_outside_bins():
    table = make_table()

    extra = pd.DataFrame(
        {
            "lead_time_hours": [
                -6.0,
                246.0,
            ],
            "track_error_km": [
                9999.0,
                9999.0,
            ],
            "pressure_error_pa": [
                9999.0,
                9999.0,
            ],
            "wind_error_ms": [
                9999.0,
                9999.0,
            ],
        }
    )

    combined = pd.concat(
        [
            table,
            extra,
        ],
        ignore_index=True,
    )

    result = summarize_lead_time_verification(
        combined
    )

    assert result[
        "point_count"
    ].sum() == len(
        table
    )


def test_summarize_empty_after_binning():
    table = pd.DataFrame(
        {
            "lead_time_hours": [
                -6.0,
                246.0,
            ],
            "track_error_km": [
                100.0,
                200.0,
            ],
            "pressure_error_pa": [
                1000.0,
                2000.0,
            ],
            "wind_error_ms": [
                10.0,
                20.0,
            ],
        }
    )

    result = summarize_lead_time_verification(
        table
    )

    assert result.empty

    assert list(
        result.columns
    ) == [
        "lead_time_bin",
        "lead_time_start_hours",
        "lead_time_end_hours",
        "point_count",
        "mean_track_error_km",
        "rmse_track_error_km",
        "median_track_error_km",
        "pressure_mae_pa",
        "pressure_rmse_pa",
        "wind_mae_ms",
        "wind_rmse_ms",
    ]


def test_summarize_invalid_input_type():
    with pytest.raises(
        TypeError,
        match="pandas DataFrame",
    ):
        summarize_lead_time_verification(
            []
        )


def test_summarize_missing_required_columns():
    table = pd.DataFrame(
        {
            "lead_time_hours": [
                0.0,
            ],
            "track_error_km": [
                100.0,
            ],
        }
    )

    with pytest.raises(
        ValueError,
        match="missing required columns",
    ):
        summarize_lead_time_verification(
            table
        )


def test_invalid_empty_bins():
    with pytest.raises(
        ValueError,
        match="bins cannot be empty",
    ):
        assign_lead_time_bins(
            [
                0.0,
            ],
            bins=(),
        )


def test_invalid_bins_type():
    with pytest.raises(
        TypeError,
        match="bins must be a tuple",
    ):
        assign_lead_time_bins(
            [
                0.0,
            ],
            bins=[
                (0.0, 24.0),
            ],
        )


def test_invalid_bin_shape():
    with pytest.raises(
        TypeError,
        match="two-element tuple",
    ):
        assign_lead_time_bins(
            [
                0.0,
            ],
            bins=(
                (0.0, 24.0, 48.0),
            ),
        )


def test_invalid_negative_bin_start():
    with pytest.raises(
        ValueError,
        match="cannot be negative",
    ):
        assign_lead_time_bins(
            [
                0.0,
            ],
            bins=(
                (-24.0, 0.0),
            ),
        )


def test_invalid_reversed_bin():
    with pytest.raises(
        ValueError,
        match="start must be less",
    ):
        assign_lead_time_bins(
            [
                0.0,
            ],
            bins=(
                (24.0, 0.0),
            ),
        )


def test_invalid_overlapping_bins():
    with pytest.raises(
        ValueError,
        match="cannot overlap",
    ):
        assign_lead_time_bins(
            [
                0.0,
            ],
            bins=(
                (0.0, 48.0),
                (24.0, 72.0),
            ),
        )

from types import SimpleNamespace


def _make_verification_table(
    lead_times,
    track_errors,
    pressure_errors=None,
    wind_errors=None,
):
    n = len(lead_times)

    if pressure_errors is None:
        pressure_errors = [
            100.0
        ] * n

    if wind_errors is None:
        wind_errors = [
            1.0
        ] * n

    return pd.DataFrame(
        {
            "lead_time_hours":
                lead_times,
            "track_error_km":
                track_errors,
            "pressure_error_pa":
                pressure_errors,
            "wind_error_ms":
                wind_errors,
        }
    )


def _make_batch_case(
    forecast_path,
):
    return SimpleNamespace(
        forecast_path=forecast_path,
    )


def _make_pipeline_result(
    **tracker_tables,
):
    verifications = {}

    for (
        tracker_name,
        table,
    ) in tracker_tables.items():

        verifications[
            tracker_name
        ] = SimpleNamespace(
            table=table
        )

    return SimpleNamespace(
        verification=SimpleNamespace(
            verifications=verifications
        )
    )


def test_batch_lead_time_combines_cases():
    cases = [
        _make_batch_case(
            "case1.zarr"
        ),
        _make_batch_case(
            "case2.zarr"
        ),
    ]

    results = [
        _make_pipeline_result(
            native=_make_verification_table(
                [0.0, 6.0],
                [100.0, 200.0],
            )
        ),
        _make_pipeline_result(
            native=_make_verification_table(
                [12.0, 18.0],
                [300.0, 400.0],
            )
        ),
    ]

    summary = pd.DataFrame(
        {
            "forecast_path": [
                "case1.zarr",
                "case2.zarr",
            ],
            "model_name": [
                "graphcast",
                "graphcast",
            ],
            "tracker": [
                "native",
                "native",
            ],
            "coverage": [
                "full",
                "full",
            ],
            "valid_for_aggregation": [
                True,
                True,
            ],
        }
    )

    aggregated = (
        aggregate_batch_lead_time_verification(
            cases,
            results,
            summary,
        )
    )

    row = aggregated.iloc[0]

    assert row[
        "lead_time_bin"
    ] == "0-24h"

    assert row[
        "case_count"
    ] == 2

    assert row[
        "point_count"
    ] == 4

    assert row[
        "mean_track_error_km"
    ] == pytest.approx(
        250.0
    )


def test_batch_lead_time_separates_trackers():
    cases = [
        _make_batch_case(
            "case1.zarr"
        )
    ]

    results = [
        _make_pipeline_result(
            native=_make_verification_table(
                [0.0, 6.0],
                [100.0, 200.0],
            ),
            wuduan=_make_verification_table(
                [0.0, 6.0],
                [50.0, 75.0],
            ),
        )
    ]

    summary = pd.DataFrame(
        {
            "forecast_path": [
                "case1.zarr",
                "case1.zarr",
            ],
            "model_name": [
                "graphcast",
                "graphcast",
            ],
            "tracker": [
                "native",
                "wuduan",
            ],
            "coverage": [
                "full",
                "full",
            ],
            "valid_for_aggregation": [
                True,
                True,
            ],
        }
    )

    aggregated = (
        aggregate_batch_lead_time_verification(
            cases,
            results,
            summary,
        )
    )

    assert set(
        aggregated["tracker"]
    ) == {
        "native",
        "wuduan",
    }


def test_batch_lead_time_excludes_failed_qc():
    cases = [
        _make_batch_case(
            "case1.zarr"
        )
    ]

    results = [
        _make_pipeline_result(
            native=_make_verification_table(
                [0.0],
                [100.0],
            ),
            vitart=_make_verification_table(
                [0.0],
                [10000.0],
            ),
        )
    ]

    summary = pd.DataFrame(
        {
            "forecast_path": [
                "case1.zarr",
                "case1.zarr",
            ],
            "model_name": [
                "graphcast",
                "graphcast",
            ],
            "tracker": [
                "native",
                "vitart",
            ],
            "coverage": [
                "full",
                "full",
            ],
            "valid_for_aggregation": [
                True,
                False,
            ],
        }
    )

    aggregated = (
        aggregate_batch_lead_time_verification(
            cases,
            results,
            summary,
        )
    )

    assert list(
        aggregated["tracker"]
    ) == [
        "native"
    ]


def test_batch_lead_time_uses_full_qc_only():
    cases = [
        _make_batch_case(
            "case1.zarr"
        )
    ]

    results = [
        _make_pipeline_result(
            native=_make_verification_table(
                [0.0],
                [100.0],
            )
        )
    ]

    summary = pd.DataFrame(
        {
            "forecast_path": [
                "case1.zarr",
                "case1.zarr",
            ],
            "model_name": [
                "graphcast",
                "graphcast",
            ],
            "tracker": [
                "native",
                "native",
            ],
            "coverage": [
                "full",
                "common",
            ],
            "valid_for_aggregation": [
                True,
                False,
            ],
        }
    )

    aggregated = (
        aggregate_batch_lead_time_verification(
            cases,
            results,
            summary,
        )
    )

    assert not aggregated.empty

    assert aggregated.iloc[0][
        "point_count"
    ] == 1


def test_batch_lead_time_counts_cases_per_bin():
    cases = [
        _make_batch_case(
            "case1.zarr"
        ),
        _make_batch_case(
            "case2.zarr"
        ),
    ]

    results = [
        _make_pipeline_result(
            native=_make_verification_table(
                [
                    0.0,
                    30.0,
                ],
                [
                    100.0,
                    200.0,
                ],
            )
        ),
        _make_pipeline_result(
            native=_make_verification_table(
                [
                    6.0,
                ],
                [
                    300.0,
                ],
            )
        ),
    ]

    summary = pd.DataFrame(
        {
            "forecast_path": [
                "case1.zarr",
                "case2.zarr",
            ],
            "model_name": [
                "graphcast",
                "graphcast",
            ],
            "tracker": [
                "native",
                "native",
            ],
            "coverage": [
                "full",
                "full",
            ],
            "valid_for_aggregation": [
                True,
                True,
            ],
        }
    )

    aggregated = (
        aggregate_batch_lead_time_verification(
            cases,
            results,
            summary,
        )
    )

    first = aggregated[
        aggregated[
            "lead_time_bin"
        ].eq(
            "0-24h"
        )
    ].iloc[0]

    second = aggregated[
        aggregated[
            "lead_time_bin"
        ].eq(
            "24-48h"
        )
    ].iloc[0]

    assert first[
        "case_count"
    ] == 2

    assert first[
        "point_count"
    ] == 2

    assert second[
        "case_count"
    ] == 1

    assert second[
        "point_count"
    ] == 1


def test_batch_lead_time_boundary_assignment():
    cases = [
        _make_batch_case(
            "case1.zarr"
        )
    ]

    results = [
        _make_pipeline_result(
            native=_make_verification_table(
                [
                    0.0,
                    24.0,
                    48.0,
                    72.0,
                    120.0,
                    168.0,
                    240.0,
                ],
                [
                    10.0,
                    20.0,
                    30.0,
                    40.0,
                    50.0,
                    60.0,
                    70.0,
                ],
            )
        )
    ]

    summary = pd.DataFrame(
        {
            "forecast_path": [
                "case1.zarr"
            ],
            "model_name": [
                "graphcast"
            ],
            "tracker": [
                "native"
            ],
            "coverage": [
                "full"
            ],
            "valid_for_aggregation": [
                True
            ],
        }
    )

    aggregated = (
        aggregate_batch_lead_time_verification(
            cases,
            results,
            summary,
        )
    )

    counts = dict(
        zip(
            aggregated[
                "lead_time_bin"
            ],
            aggregated[
                "point_count"
            ],
        )
    )

    assert counts == {
        "0-24h": 1,
        "24-48h": 1,
        "48-72h": 1,
        "72-120h": 1,
        "120-168h": 1,
        "168-240h": 2,
    }


def test_batch_lead_time_pooled_rmse():
    cases = [
        _make_batch_case(
            "case1.zarr"
        ),
        _make_batch_case(
            "case2.zarr"
        ),
    ]

    results = [
        _make_pipeline_result(
            native=_make_verification_table(
                [0.0],
                [3.0],
            )
        ),
        _make_pipeline_result(
            native=_make_verification_table(
                [6.0],
                [4.0],
            )
        ),
    ]

    summary = pd.DataFrame(
        {
            "forecast_path": [
                "case1.zarr",
                "case2.zarr",
            ],
            "model_name": [
                "graphcast",
                "graphcast",
            ],
            "tracker": [
                "native",
                "native",
            ],
            "coverage": [
                "full",
                "full",
            ],
            "valid_for_aggregation": [
                True,
                True,
            ],
        }
    )

    aggregated = (
        aggregate_batch_lead_time_verification(
            cases,
            results,
            summary,
        )
    )

    row = aggregated.iloc[0]

    expected = np.sqrt(
        (
            3.0 ** 2
            + 4.0 ** 2
        )
        / 2.0
    )

    assert row[
        "rmse_track_error_km"
    ] == pytest.approx(
        expected
    )