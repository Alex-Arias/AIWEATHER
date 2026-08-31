import json
from datetime import datetime

import numpy as np
import pandas as pd

from aiweather.forecast import ForecastMetadata
from aiweather.tracking import (
    TrackRecord,
    export_operational_track,
)


def test_export_operational_track(tmp_path):
    records = [
        TrackRecord(
            lead_time_hours=0,
            valid_time=np.datetime64(
                "2026-08-29T12:00:00"
            ),
            latitude=15.5,
            longitude=243.0,
            pressure=98460.0,
            pressure_units="Pa",
            max_wind=28.5,
            wind_units="m/s",
        ),
        TrackRecord(
            lead_time_hours=6,
            valid_time=np.datetime64(
                "2026-08-29T18:00:00"
            ),
            latitude=16.0,
            longitude=242.0,
            pressure=98200.0,
            pressure_units="Pa",
            max_wind=29.0,
            wind_units="m/s",
            distance_km=120.0,
            translation_speed_kmh=20.0,
            bearing_degrees=300.0,
            cumulative_distance_km=120.0,
        ),
    ]

    metadata = ForecastMetadata(
        model_name="graphcast",
        model_version="unknown",
        backend="earth2studio",
        forecast_id=(
            "graphcast_gfs_"
            "20260829T120000_240h"
        ),
        initialization_time=datetime(
            2026, 8, 29, 12
        ),
    )

    track_path, provenance_path = (
        export_operational_track(
            records,
            tmp_path,
            filename="graphcast_track.csv",
            forecast_path=(
                "outputs/graphcast/"
                "20260829T120000/"
                "forecast.zarr"
            ),
            forecast_metadata=metadata,
            storm_id="EP112026",
            storm_name="Karina",
            seed_latitude=15.3,
            seed_longitude=-117.2,
            seed_valid_time=datetime(
                2026, 8, 29, 12
            ),
            seed_source=(
                "NHC Forecast/Advisory #8"
            ),
            seed_source_issuance_time=datetime(
                2026, 8, 29, 15
            ),
            experiment_type=(
                "pseudo-operational-"
                "retrospective-seed"
            ),
            lat_min=5.0,
            lat_max=40.0,
            lon_min=-160.0,
            lon_max=-90.0,
            search_radius_km=500.0,
            wind_radius_km=300.0,
            maximum_translation_speed_mps=20.0,
            datasource="gfs",
            git_commit="cbfdc71",
        )
    )

    assert track_path == (
        tmp_path / "graphcast_track.csv"
    )
    assert provenance_path == (
        tmp_path
        / "graphcast_track.provenance.json"
    )

    dataframe = pd.read_csv(track_path)

    assert len(dataframe) == 2
    assert list(dataframe["lead_time_hours"]) == [
        0,
        6,
    ]
    assert dataframe.iloc[-1]["latitude"] == 16.0

    with provenance_path.open(
        encoding="utf-8"
    ) as handle:
        manifest = json.load(handle)

    assert manifest["schema_version"] == 1
    assert manifest["storm"]["id"] == "EP112026"
    assert manifest["storm"]["name"] == "Karina"

    assert (
        manifest["forecast"]["model_name"]
        == "graphcast"
    )
    assert (
        manifest["forecast"]["datasource"]
        == "gfs"
    )

    assert manifest["seed"]["latitude"] == 15.3
    assert (
        manifest["seed"]["source_issuance_time"]
        == "2026-08-29T15:00:00"
    )

    tracking = manifest["tracking"]

    assert tracking["number_of_points"] == 2
    assert tracking["last_lead_time_hours"] == 6
    assert (
        tracking["last_valid_time"]
        == "2026-08-29T18:00:00"
    )
    assert tracking["last_latitude"] == 16.0
    assert tracking["last_longitude"] == 242.0
    assert (
        tracking[
            "maximum_translation_speed_mps"
        ]
        == 20.0
    )

    assert (
        manifest["software"]["git_commit"]
        == "cbfdc71"
    )


def test_export_operational_track_empty_records(
    tmp_path,
):
    metadata = ForecastMetadata(
        model_name="graphcast",
        model_version="unknown",
        backend="earth2studio",
        initialization_time=datetime(
            2026, 8, 29, 12
        ),
    )

    _, provenance_path = export_operational_track(
        [],
        tmp_path,
        filename="empty_track.csv",
        forecast_path="forecast.zarr",
        forecast_metadata=metadata,
        storm_id="EP112026",
        storm_name="Karina",
        seed_latitude=15.3,
        seed_longitude=-117.2,
        seed_valid_time=datetime(
            2026, 8, 29, 12
        ),
        seed_source="test",
        seed_source_issuance_time=None,
        experiment_type="test",
        lat_min=5.0,
        lat_max=40.0,
        lon_min=-160.0,
        lon_max=-90.0,
        search_radius_km=500.0,
        wind_radius_km=300.0,
        maximum_translation_speed_mps=None,
    )

    with provenance_path.open(
        encoding="utf-8"
    ) as handle:
        manifest = json.load(handle)

    tracking = manifest["tracking"]

    assert tracking["number_of_points"] == 0
    assert tracking["last_lead_time_hours"] is None
    assert tracking["last_valid_time"] is None
    assert tracking["last_latitude"] is None
    assert tracking["last_longitude"] is None
    assert (
        tracking[
            "maximum_translation_speed_mps"
        ]
        is None
    )
