from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "compare_operational_tc_cycles.py"
)

SPEC = importlib.util.spec_from_file_location(
    "compare_operational_tc_cycles",
    SCRIPT_PATH,
)

MODULE = importlib.util.module_from_spec(SPEC)

assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def _track(
    init: str,
    leads: list[int],
    times: list[str],
    latitudes: list[float],
    longitudes: list[float],
):
    dataframe = pd.DataFrame(
        {
            "lead_time_hours": leads,
            "valid_time": times,
            "latitude": latitudes,
            "longitude": longitudes,
        }
    )

    dataframe["valid_time"] = pd.to_datetime(
        dataframe["valid_time"]
    )

    return {
        "dataframe": dataframe,
        "provenance": {
            "storm": {
                "id": "EP132026",
                "name": "Marie",
            },
            "forecast": {
                "initialization_time": init,
            },
        },
    }


def test_cycle_displacement_uses_common_valid_times():
    baseline = {
        "model_a": _track(
            "2026-09-03T00:00:00",
            [48, 54],
            [
                "2026-09-05T00:00:00",
                "2026-09-05T06:00:00",
            ],
            [20.0, 21.0],
            [-120.0, -121.0],
        )
    }

    later = {
        "model_a": _track(
            "2026-09-05T00:00:00",
            [0, 6],
            [
                "2026-09-05T00:00:00",
                "2026-09-05T06:00:00",
            ],
            [20.5, 22.0],
            [-120.0, -121.0],
        )
    }

    detailed, summary = (
        MODULE.build_cycle_displacement(
            baseline,
            later,
        )
    )

    assert len(detailed) == 2
    assert detailed[
        "lead_time_hours_baseline"
    ].tolist() == [48, 54]
    assert detailed[
        "lead_time_hours_later"
    ].tolist() == [0, 6]
    assert detailed[
        "delta_lead_hours"
    ].tolist() == [48, 48]

    assert detailed[
        "delta_latitude_degrees"
    ].tolist() == [0.5, 1.0]

    assert (
        detailed["cycle_displacement_km"] > 0.0
    ).all()

    assert summary.loc[
        0,
        "number_of_common_times",
    ] == 2


def test_longitude_difference_wraps_dateline():
    baseline = {
        "model_a": _track(
            "2026-09-03T00:00:00",
            [48],
            ["2026-09-05T00:00:00"],
            [20.0],
            [179.0],
        )
    }

    later = {
        "model_a": _track(
            "2026-09-05T00:00:00",
            [0],
            ["2026-09-05T00:00:00"],
            [20.0],
            [-179.0],
        )
    }

    detailed, _ = MODULE.build_cycle_displacement(
        baseline,
        later,
    )

    assert detailed.loc[
        0,
        "delta_longitude_degrees",
    ] == 2.0


def test_resolve_spread_models_defaults_to_all_common():
    baseline = {
        "aifs2": {},
        "graphcast": {},
        "pangu3": {},
        "pangu6": {},
    }
    later = dict(baseline)

    assert MODULE.resolve_spread_models(
        baseline,
        later,
    ) == [
        "aifs2",
        "graphcast",
        "pangu3",
        "pangu6",
    ]


def test_resolve_spread_models_uses_requested_subset():
    baseline = {
        "aifs2": {},
        "graphcast": {},
        "pangu3": {},
        "pangu6": {},
    }
    later = dict(baseline)

    assert MODULE.resolve_spread_models(
        baseline,
        later,
        ["aifs2", "graphcast", "pangu3"],
    ) == [
        "aifs2",
        "graphcast",
        "pangu3",
    ]


def test_system_spread_uses_requested_systems():
    tracks = {
        "aifs2": _track(
            "2026-09-05T00:00:00",
            [0],
            ["2026-09-05T00:00:00"],
            [20.0],
            [-120.0],
        ),
        "graphcast": _track(
            "2026-09-05T00:00:00",
            [0],
            ["2026-09-05T00:00:00"],
            [21.0],
            [-120.0],
        ),
        "pangu3": _track(
            "2026-09-05T00:00:00",
            [0],
            ["2026-09-05T00:00:00"],
            [22.0],
            [-120.0],
        ),
        "pangu6": _track(
            "2026-09-05T00:00:00",
            [0],
            ["2026-09-05T00:00:00"],
            [40.0],
            [-120.0],
        ),
    }

    models = [
        "aifs2",
        "graphcast",
        "pangu3",
    ]

    spread = MODULE.build_system_spread(
        tracks,
        "later",
        models,
    )

    assert len(spread) == 1
    assert spread.loc[
        0, "number_of_systems"
    ] == 3
    assert spread.loc[
        0, "systems"
    ] == "aifs2,graphcast,pangu3"
    assert spread.loc[
        0, "maximum_pairwise_spread_km"
    ] < 300.0


def test_match_spread_uses_same_valid_times():
    baseline = pd.DataFrame(
        {
            "cycle": ["baseline"] * 3,
            "valid_time": pd.to_datetime(
                [
                    "2026-09-05T00:00:00",
                    "2026-09-05T06:00:00",
                    "2026-09-05T12:00:00",
                ]
            ),
            "mean_pairwise_spread_km":
                [100.0, 200.0, 300.0],
        }
    )

    later = pd.DataFrame(
        {
            "cycle": ["later"] * 2,
            "valid_time": pd.to_datetime(
                [
                    "2026-09-05T00:00:00",
                    "2026-09-05T06:00:00",
                ]
            ),
            "mean_pairwise_spread_km":
                [80.0, 120.0],
        }
    )

    baseline_matched, later_matched = (
        MODULE.match_spread_valid_times(
            baseline,
            later,
        )
    )

    assert len(baseline_matched) == 2
    assert len(later_matched) == 2

    assert baseline_matched[
        "valid_time"
    ].tolist() == later_matched[
        "valid_time"
    ].tolist()

    assert baseline_matched[
        "mean_pairwise_spread_km"
    ].tolist() == [100.0, 200.0]

    assert later_matched[
        "mean_pairwise_spread_km"
    ].tolist() == [80.0, 120.0]
