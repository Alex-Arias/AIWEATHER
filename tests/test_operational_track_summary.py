import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import xarray as xr

import scripts.summarize_operational_tracks as summary


def test_summarize_track_pairs_provenance_with_csv(
    tmp_path,
    monkeypatch,
):
    forecast_path = tmp_path / "forecast.zarr"

    dataset = xr.Dataset(
        coords={
            "lead_time": np.array(
                [
                    np.timedelta64(0, "h"),
                    np.timedelta64(6, "h"),
                    np.timedelta64(12, "h"),
                ]
            )
        }
    )

    fake_forecast = SimpleNamespace(
        dataset=dataset
    )

    monkeypatch.setattr(
        summary,
        "open_forecast",
        lambda path: fake_forecast,
    )

    track_path = tmp_path / "test_track.csv"

    pd.DataFrame(
        {
            "lead_time_hours": [0, 6, 12],
            "valid_time": [
                "2026-08-29 12:00:00",
                "2026-08-29 18:00:00",
                "2026-08-30 00:00:00",
            ],
            "latitude": [15.3, 15.5, 16.0],
            "longitude": [242.8, 242.5, 242.0],
        }
    ).to_csv(
        track_path,
        index=False,
    )

    provenance_path = (
        tmp_path
        / "test_track.provenance.json"
    )

    provenance = {
        "storm": {
            "id": "EP112026",
            "name": "Karina",
        },
        "forecast": {
            "model_name": "testmodel",
            "forecast_id": "test_forecast",
            "path": str(forecast_path),
            "datasource": "gfs",
            "datasource_source": None,
            "initialization_time": (
                "2026-08-29T12:00:00"
            ),
        },
        "seed": {
            "latitude": 15.3,
            "longitude": -117.2,
            "valid_time": (
                "2026-08-29T12:00:00"
            ),
            "source": "NHC",
            "source_issuance_time": (
                "2026-08-29T15:00:00"
            ),
        },
        "tracking": {
            "experiment_type": (
                "pseudo-operational"
            ),
            "search_radius_km": 500.0,
            "wind_radius_km": 300.0,
            "maximum_translation_speed_mps": (
                20.0
            ),
        },
        "software": {
            "git_commit": "abc123",
        },
    }

    provenance_path.write_text(
        json.dumps(provenance)
    )

    result = summary.summarize_track(
        provenance_path
    )

    assert result["model"] == "testmodel"
    assert result["number_of_points"] == 3
    assert result["first_lead_time_hours"] == 0
    assert result["last_lead_time_hours"] == 12
    assert result["forecast_horizon_hours"] == 12
    assert result["reaches_forecast_horizon"] is True
    assert result["last_latitude"] == 16.0
    assert result["last_longitude"] == 242.0


def test_main_discovers_tracks_in_model_subdirectories(
    tmp_path,
    monkeypatch,
):
    model_dir = tmp_path / "model_a"
    model_dir.mkdir()

    provenance_path = (
        model_dir
        / "model_a_track.provenance.json"
    )
    provenance_path.write_text(
        "{}",
        encoding="utf-8",
    )

    seen_paths = []

    def fake_summarize_track(path):
        seen_paths.append(path)
        return {
            "model": "model_a",
            "number_of_points": 3,
            "last_lead_time_hours": 12,
            "forecast_horizon_hours": 12,
            "reaches_forecast_horizon": True,
        }

    monkeypatch.setattr(
        summary,
        "summarize_track",
        fake_summarize_track,
    )

    monkeypatch.setattr(
        "sys.argv",
        [
            "summarize_operational_tracks.py",
            "--input-dir",
            str(tmp_path),
        ],
    )

    summary.main()

    assert seen_paths == [
        provenance_path
    ]
    assert (
        tmp_path / "track_summary.csv"
    ).exists()
