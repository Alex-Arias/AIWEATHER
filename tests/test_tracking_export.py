import json
from datetime import datetime

import numpy as np
import pandas as pd

from aiweather.forecast import ForecastMetadata
from aiweather.tracking import (
    GenesisResult,
    TrackRecord,
    export_native_genesis_case,
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

def test_export_native_genesis_case(tmp_path):
    genesis = GenesisResult(
        track_index=2,
        genesis_lead_time_hours=72,
        latitude=15.0,
        longitude=257.5,
        pressure=99332.0,
        max_wind=18.7,
        qualifying_points=6,
    )

    records = [
        TrackRecord(
            lead_time_hours=72,
            valid_time=np.datetime64(
                "2026-09-23T12:00:00"
            ),
            latitude=15.0,
            longitude=257.5,
            pressure=99332.0,
            pressure_units="Pa",
            max_wind=18.7,
            wind_units="m/s",
        ),
        TrackRecord(
            lead_time_hours=78,
            valid_time=np.datetime64(
                "2026-09-23T18:00:00"
            ),
            latitude=15.25,
            longitude=257.0,
            pressure=99250.0,
            pressure_units="Pa",
            max_wind=19.2,
            wind_units="m/s",
        ),
    ]

    metadata = ForecastMetadata(
        model_name="pangu6",
        model_version="unknown",
        backend="earth2studio",
        forecast_id=(
            "pangu6_gfs_"
            "20260920T120000_240h"
        ),
        initialization_time=datetime(
            2026, 9, 20, 12
        ),
    )

    track_path, provenance_path = (
        export_native_genesis_case(
            genesis,
            records,
            tmp_path,
            filename="pangu6_track.csv",
            forecast_path="forecast.zarr",
            forecast_metadata=metadata,
            case_id=(
                "epac_candidate_"
                "20260920T120000"
            ),
            case_name=None,
            experiment_type=(
                "prospective-genesis"
            ),
            lat_min=5.0,
            lat_max=25.0,
            lon_min=-115.0,
            lon_max=-95.0,
            candidate_lead_min_hours=24,
            candidate_lead_max_hours=174,
            candidate_interval_hours=6,
            max_candidates=5,
            minimum_separation_km=500.0,
            maximum_displacement_km=500.0,
            minimum_wind=17.0,
            maximum_pressure=100500.0,
            minimum_consecutive_points=3,
            wind_radius_km=300.0,
            search_radius_km=500.0,
            datasource="gfs",
            datasource_source="aws",
            git_commit="testcommit",
        )
    )

    dataframe = pd.read_csv(track_path)

    assert len(dataframe) == 2
    assert (
        list(dataframe["lead_time_hours"])
        == [72, 78]
    )

    with provenance_path.open(
        encoding="utf-8"
    ) as handle:
        manifest = json.load(handle)

    assert manifest["schema_version"] == 1
    assert (
        manifest["experiment_type"]
        == "prospective-genesis"
    )
    assert (
        manifest["case"]["id"]
        == "epac_candidate_20260920T120000"
    )

    detection = manifest[
        "genesis_detection"
    ]

    assert (
        detection[
            "candidate_lead_min_hours"
        ]
        == 24
    )
    assert (
        detection[
            "candidate_lead_max_hours"
        ]
        == 174
    )
    assert (
        detection["result"][
            "genesis_lead_time_hours"
        ]
        == 72
    )
    assert (
        detection["result"][
            "qualifying_points"
        ]
        == 6
    )

    tracking = manifest[
        "post_genesis_tracking"
    ]

    assert tracking["number_of_points"] == 2
    assert tracking["last_lead_time_hours"] == 78

    assert (
        manifest["software"]["git_commit"]
        == "testcommit"
    )
