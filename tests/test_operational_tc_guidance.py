import pandas as pd
import pytest

from scripts.build_operational_tc_guidance import (
    availability_status,
    build_guidance,
    model_family,
    parse_bool_series,
)


def variability_frame():
    return pd.DataFrame(
        {
            "valid_time": [
                "2026-09-28 12:00:00",
                "2026-09-28 18:00:00",
            ],
            "consensus_latitude": [20.0, 21.0],
            "consensus_longitude": [-115.0, -114.0],
            "n_models": [4, 2],
            "r67_km": [10.0, 12.0],
            "r90_km": [20.0, 15.0],
            "max_distance_km": [25.0, 18.0],
            "max_lead_h": [0, 6],
            "spread_valid": [True, True],
        }
    )


def member_frame():
    rows = []

    cycle = "2026-09-28 12:00:00"

    for model in (
        "aifs2",
        "graphcast",
        "pangu3",
        "pangu6",
    ):
        rows.append(
            {
                "valid_time":
                    "2026-09-28 12:00:00",
                "cycle_time": cycle,
                "model": model,
            }
        )

    for model in (
        "aifs2",
        "graphcast",
    ):
        rows.append(
            {
                "valid_time":
                    "2026-09-28 18:00:00",
                "cycle_time": cycle,
                "model": model,
            }
        )

    return pd.DataFrame(rows)


def test_model_family_mapping():
    assert model_family("aifs2") == "AIFS2"
    assert model_family("graphcast") == "GraphCast"
    assert model_family("pangu3") == "Pangu-Weather"
    assert model_family("pangu6") == "Pangu-Weather"

    with pytest.raises(ValueError):
        model_family("unknown-model")


def test_availability_status_uses_family_count():
    assert availability_status(3) == "SUPPORTED"
    assert availability_status(2) == "LIMITED"
    assert availability_status(1) == "INSUFFICIENT"


def test_strict_boolean_parser():
    result = parse_bool_series(
        pd.Series(["True", "False", "1", "0"])
    )

    assert result.tolist() == [
        True,
        False,
        True,
        False,
    ]

    with pytest.raises(
        ValueError,
        match="Invalid boolean",
    ):
        parse_bool_series(
            pd.Series(["True", "not-a-bool"])
        )


def test_guidance_separates_configurations_and_families():
    result = build_guidance(
        variability_frame(),
        member_frame(),
        storm="Test Storm",
    )

    assert list(result["n_configurations"]) == [4, 2]
    assert list(result["n_model_families"]) == [3, 2]

    assert list(result["availability_status"]) == [
        "SUPPORTED",
        "LIMITED",
    ]

    assert list(result["guidance_supported"]) == [
        True,
        False,
    ]


def test_false_spread_remains_unsupported():
    variability = variability_frame()
    variability.loc[0, "spread_valid"] = False

    result = build_guidance(
        variability,
        member_frame(),
        storm="Test Storm",
    )

    assert not bool(
        result.loc[0, "guidance_supported"]
    )


def test_configuration_mismatch_raises():
    variability = variability_frame()
    variability.loc[0, "n_models"] = 3

    with pytest.raises(
        ValueError,
        match="configuration-count mismatch",
    ):
        build_guidance(
            variability,
            member_frame(),
            storm="Test Storm",
        )
