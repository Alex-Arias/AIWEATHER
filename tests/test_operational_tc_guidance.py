import pandas as pd
import pytest

from scripts.build_operational_tc_guidance import (
    availability_status,
    build_guidance,
)


def sample_variability():
    return pd.DataFrame(
        {
            "valid_time": [
                "2026-09-28 12:00:00",
                "2026-09-28 18:00:00",
                "2026-09-29 00:00:00",
            ],
            "consensus_latitude": [20.0, 21.0, 22.0],
            "consensus_longitude": [-120.0, -119.0, -118.0],
            "n_models": [4, 3, 2],
            "r67_km": [10.0, 20.0, 15.0],
            "r90_km": [20.0, 40.0, 25.0],
            "max_distance_km": [25.0, 45.0, 30.0],
            "min_lead_h": [0, 6, 12],
            "max_lead_h": [0, 6, 12],
            "spread_valid": [True, True, True],
        }
    )


def test_availability_status():
    assert availability_status(4) == "SUPPORTED"
    assert availability_status(3) == "SUPPORTED"
    assert availability_status(2) == "LIMITED"
    assert availability_status(1) == "INSUFFICIENT"


def test_build_guidance_separates_availability_from_spread():
    result = build_guidance(
        sample_variability(),
        storm="Test Storm",
        tracker="wuduan",
    )

    assert list(result["availability_status"]) == [
        "SUPPORTED",
        "SUPPORTED",
        "LIMITED",
    ]

    assert list(result["guidance_supported"]) == [
        True,
        True,
        False,
    ]


def test_invalid_spread_is_not_supported():
    data = sample_variability()
    data.loc[0, "spread_valid"] = False

    result = build_guidance(
        data,
        storm="Test Storm",
    )

    assert not bool(result.loc[0, "guidance_supported"])


def test_metadata_and_lead_time_are_preserved():
    result = build_guidance(
        sample_variability(),
        storm="Polo",
        tracker="wuduan",
    )

    assert set(result["storm"]) == {"Polo"}
    assert set(result["tracker"]) == {"wuduan"}
    assert list(result["lead_time_hours"]) == [0, 6, 12]


def test_missing_required_column_raises():
    data = sample_variability().drop(
        columns=["r90_km"]
    )

    with pytest.raises(
        ValueError,
        match="r90_km",
    ):
        build_guidance(
            data,
            storm="Polo",
        )
