from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "compare_operational_tc_tracks.py"
)

SPEC = importlib.util.spec_from_file_location(
    "compare_operational_tc_tracks",
    SCRIPT_PATH,
)

MODULE = importlib.util.module_from_spec(
    SPEC
)

assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def _write_track(
    directory: Path,
    *,
    model: str,
    longitudes: list[float],
) -> None:
    track_path = (
        directory
        / f"{model}_track.csv"
    )

    provenance_path = (
        directory
        / f"{model}_track.provenance.json"
    )

    dataframe = pd.DataFrame(
        {
            "lead_time_hours": [0, 6, 12],
            "valid_time": [
                "2026-08-29T12:00:00",
                "2026-08-29T18:00:00",
                "2026-08-30T00:00:00",
            ],
            "latitude": [
                15.0,
                16.0,
                17.0,
            ],
            "longitude": longitudes,
            "pressure": [
                100000.0,
                99000.0,
                98000.0,
            ],
            "pressure_units": [
                "Pa",
                "Pa",
                "Pa",
            ],
            "max_wind": [
                10.0,
                15.0,
                20.0,
            ],
            "wind_units": [
                "m/s",
                "m/s",
                "m/s",
            ],
            "distance_km": [
                np.nan,
                100.0,
                100.0,
            ],
            "translation_speed_kmh": [
                np.nan,
                16.7,
                16.7,
            ],
            "bearing_degrees": [
                np.nan,
                270.0,
                270.0,
            ],
            "cumulative_distance_km": [
                0.0,
                100.0,
                200.0,
            ],
        }
    )

    dataframe.to_csv(
        track_path,
        index=False,
    )

    provenance = {
        "schema_version": 1,
        "storm": {
            "id": "EP112026",
            "name": "Karina",
        },
        "forecast": {
            "model_name": model,
            "forecast_id": (
                f"{model}_test_"
                "20260829T120000_12h"
            ),
            "initialization_time": (
                "2026-08-29T12:00:00"
            ),
        },
        "seed": {
            "latitude": 15.3,
            "longitude": -117.2,
        },
        "tracking": {
            "experiment_type":
                "pseudo-operational",
        },
    }

    provenance_path.write_text(
        json.dumps(provenance),
        encoding="utf-8",
    )


def test_discover_tracks_and_records(
    tmp_path,
):
    _write_track(
        tmp_path,
        model="model_a",
        longitudes=[
            240.0,
            239.0,
            238.0,
        ],
    )

    tracks = MODULE.discover_tracks(
        tmp_path
    )

    assert set(tracks) == {
        "model_a"
    }

    records = tracks[
        "model_a"
    ]["records"]

    assert len(records) == 3
    assert records[0].lead_time_hours == 0
    assert records[-1].lead_time_hours == 12
    assert records[-1].latitude == 17.0
    assert records[-1].longitude == 238.0
    assert records[-1].pressure == 98000.0
    assert records[-1].max_wind == 20.0


def test_pairwise_comparison_uses_common_leads(
    tmp_path,
):
    _write_track(
        tmp_path,
        model="model_a",
        longitudes=[
            240.0,
            239.0,
            238.0,
        ],
    )

    _write_track(
        tmp_path,
        model="model_b",
        longitudes=[
            240.0,
            238.0,
            236.0,
        ],
    )

    tracks = MODULE.discover_tracks(
        tmp_path
    )

    detailed, summary = (
        MODULE.build_pairwise_comparisons(
            tracks
        )
    )

    assert len(detailed) == 3

    assert detailed[
        "lead_time_hours"
    ].tolist() == [
        0,
        6,
        12,
    ]

    assert (
        "track_separation_km"
        in detailed.columns
    )

    assert (
        "track_error_km"
        not in detailed.columns
    )

    assert detailed.loc[
        0,
        "track_separation_km",
    ] == 0.0

    assert summary.loc[
        0,
        "number_of_common_times",
    ] == 3

    assert summary.loc[
        0,
        "first_common_lead_hours",
    ] == 0

    assert summary.loc[
        0,
        "last_common_lead_hours",
    ] == 12

    assert summary.loc[
        0,
        "maximum_separation_km",
    ] > 0.0


def test_validate_common_case_rejects_mismatch(
    tmp_path,
):
    _write_track(
        tmp_path,
        model="model_a",
        longitudes=[
            240.0,
            239.0,
            238.0,
        ],
    )

    _write_track(
        tmp_path,
        model="model_b",
        longitudes=[
            240.0,
            239.0,
            238.0,
        ],
    )

    provenance_path = (
        tmp_path
        / "model_b_track.provenance.json"
    )

    provenance = json.loads(
        provenance_path.read_text(
            encoding="utf-8",
        )
    )

    provenance[
        "storm"
    ]["id"] = "OTHER"

    provenance_path.write_text(
        json.dumps(provenance),
        encoding="utf-8",
    )

    tracks = MODULE.discover_tracks(
        tmp_path
    )

    import pytest

    with pytest.raises(
        ValueError,
        match="different storm IDs",
    ):
        MODULE.validate_common_case(
            tracks
        )