from pathlib import Path

import numpy as np
import pandas as pd

from aiweather.tracking.evaluation import (
    TrackerEvaluation,
)
from aiweather.tracking.genesis import (
    GenesisResult,
)
from aiweather.tracking.records import (
    TrackRecord,
)
from aiweather.tracking.workflow import (
    TrackingWorkflowResult,
)
from aiweather.verification.comparison import (
    compare_forecast_to_best_track,
)
from aiweather.verification.export import (
    export_verification_case,
    export_verification_manifest,
)
from aiweather.verification.workflow import (
    VerificationWorkflowResult,
)

import json
from datetime import datetime

from aiweather.forecast.metadata import (
    ForecastMetadata,
)


def make_record(
    lead_time_hours,
    latitude,
    longitude,
    *,
    pressure=100000.0,
    max_wind=20.0,
):
    return TrackRecord(
        lead_time_hours=lead_time_hours,
        valid_time=(
            np.datetime64(
                "2026-07-24T00:00:00"
            )
            + np.timedelta64(
                lead_time_hours,
                "h",
            )
        ),
        latitude=latitude,
        longitude=longitude,
        pressure=pressure,
        pressure_units="Pa",
        max_wind=max_wind,
        wind_units="m/s",
    )


def make_genesis():
    return GenesisResult(
        track_index=0,
        genesis_lead_time_hours=54,
        latitude=13.0,
        longitude=254.0,
        pressure=100000.0,
        max_wind=20.0,
        qualifying_points=3,
    )


def test_export_verification_case_native_only(
    tmp_path,
):
    native = [
        make_record(
            54,
            13.0,
            254.0,
        ),
        make_record(
            60,
            13.5,
            253.0,
        ),
    ]

    observations = [
        make_record(
            54,
            12.6,
            -107.6,
            pressure=98000.0,
            max_wind=40.0,
        ),
        make_record(
            60,
            13.2,
            -108.8,
            pressure=97000.0,
            max_wind=45.0,
        ),
    ]

    native_verification = (
        compare_forecast_to_best_track(
            native,
            observations,
            forecast_name="native",
            observation_name="ibtracs",
        )
    )

    tracking = TrackingWorkflowResult(
        genesis=make_genesis(),
        native_records=native,
        evaluation=None,
    )

    verification = VerificationWorkflowResult(
        observations=observations,
        native=native_verification,
        wuduan=None,
        vitart=None,
    )

    output = export_verification_case(
        tmp_path / "case",
        tracking_result=tracking,
        verification_result=verification,
    )

    assert output == (
        tmp_path / "case"
    )

    expected = {
        "ibtracs.csv",
        "native.csv",
        "verification_native.csv",
        "common_native.csv",
        "verification_summary.csv",
    }

    assert {
        path.name
        for path in output.iterdir()
    } == expected

    summary = pd.read_csv(
        output / "verification_summary.csv"
    )

    assert len(summary) == 2

    assert set(
        summary["coverage"]
    ) == {
        "full",
        "common",
    }

    assert set(
        summary["tracker"]
    ) == {
        "native",
    }


def test_export_verification_case_all_trackers(
    tmp_path,
):
    native = [
        make_record(
            54,
            13.0,
            254.0,
        ),
        make_record(
            60,
            13.5,
            253.0,
        ),
    ]

    wuduan = [
        make_record(
            54,
            12.9,
            253.9,
        ),
        make_record(
            60,
            13.4,
            252.9,
        ),
    ]

    vitart = [
        make_record(
            60,
            13.5,
            253.0,
        ),
    ]

    observations = [
        make_record(
            54,
            12.6,
            -107.6,
            pressure=98000.0,
            max_wind=40.0,
        ),
        make_record(
            60,
            13.2,
            -108.8,
            pressure=97000.0,
            max_wind=45.0,
        ),
    ]

    native_verification = (
        compare_forecast_to_best_track(
            native,
            observations,
            forecast_name="native",
            observation_name="ibtracs",
        )
    )

    wuduan_verification = (
        compare_forecast_to_best_track(
            wuduan,
            observations,
            forecast_name="wuduan",
            observation_name="ibtracs",
        )
    )

    vitart_verification = (
        compare_forecast_to_best_track(
            vitart,
            observations,
            forecast_name="vitart",
            observation_name="ibtracs",
        )
    )

    wuduan_match = type(
        "DummyMatch",
        (),
        {
            "records": wuduan,
        },
    )()

    vitart_match = type(
        "DummyMatch",
        (),
        {
            "records": vitart,
        },
    )()

    evaluation = TrackerEvaluation(
        reference_records=native,
        wuduan_match=wuduan_match,
        vitart_match=vitart_match,
    )

    tracking = TrackingWorkflowResult(
        genesis=make_genesis(),
        native_records=native,
        evaluation=evaluation,
    )

    verification = VerificationWorkflowResult(
        observations=observations,
        native=native_verification,
        wuduan=wuduan_verification,
        vitart=vitart_verification,
    )

    output = export_verification_case(
        tmp_path / "case",
        tracking_result=tracking,
        verification_result=verification,
    )

    expected = {
        "ibtracs.csv",
        "native.csv",
        "wuduan.csv",
        "vitart.csv",
        "verification_native.csv",
        "verification_wuduan.csv",
        "verification_vitart.csv",
        "common_native.csv",
        "common_wuduan.csv",
        "common_vitart.csv",
        "verification_summary.csv",
    }

    assert {
        path.name
        for path in output.iterdir()
    } == expected

    summary = pd.read_csv(
        output / "verification_summary.csv"
    )

    assert len(summary) == 6

    assert set(
        summary["tracker"]
    ) == {
        "native",
        "wuduan",
        "vitart",
    }

    assert set(
        summary["coverage"]
    ) == {
        "full",
        "common",
    }

    common_native = pd.read_csv(
        output / "common_native.csv"
    )

    common_wuduan = pd.read_csv(
        output / "common_wuduan.csv"
    )

    common_vitart = pd.read_csv(
        output / "common_vitart.csv"
    )

    assert len(common_native) == 1
    assert len(common_wuduan) == 1
    assert len(common_vitart) == 1


def test_export_verification_case_invalid_tracking(
    tmp_path,
):
    try:
        export_verification_case(
            tmp_path,
            tracking_result="invalid",
            verification_result="invalid",
        )
    except TypeError as exc:
        assert "TrackingWorkflowResult" in str(
            exc
        )
    else:
        raise AssertionError(
            "Expected TypeError."
        )


def test_export_verification_case_invalid_verification(
    tmp_path,
):
    tracking = TrackingWorkflowResult(
        genesis=make_genesis(),
        native_records=[],
        evaluation=None,
    )

    try:
        export_verification_case(
            tmp_path,
            tracking_result=tracking,
            verification_result="invalid",
        )
    except TypeError as exc:
        assert (
            "VerificationWorkflowResult"
            in str(exc)
        )
    else:
        raise AssertionError(
            "Expected TypeError."
        )

def test_export_verification_manifest(
    tmp_path,
):
    metadata = ForecastMetadata(
        model_name="graphcast",
        model_version="unknown",
        backend="earth2studio",
        forecast_id=(
            "graphcast_gfs_"
            "20260724T000000_240h"
        ),
        initialization_time=datetime(
            2026,
            7,
            24,
            0,
            0,
        ),
    )

    path = export_verification_manifest(
        tmp_path,
        forecast_path=(
            "outputs/graphcast/"
            "20260724T000000/"
            "forecast.zarr"
        ),
        forecast_metadata=metadata,
        sid="2026204N08267",
        lat_min=5.0,
        lat_max=35.0,
        lon_min=-130.0,
        lon_max=-90.0,
        device="cuda",
        minimum_overlap=3,
        ibtracs_path=(
            "data/verification/ibtracs/"
            "ibtracs.EP.list.v04r01.csv"
        ),
        ibtracs_basin="EP",
        generate_plots=True,
        field_lead_times=[
            54,
            78,
            96,
            120,
        ],
    )

    assert path == (
        tmp_path
        / "run_manifest.json"
    )

    assert path.exists()

    with path.open(
        encoding="utf-8",
    ) as handle:
        manifest = json.load(
            handle
        )

    assert manifest[
        "schema_version"
    ] == 1

    assert manifest[
        "aiweather_version"
    ] == "0.1.0"

    assert manifest[
        "forecast"
    ][
        "model_name"
    ] == "graphcast"

    assert manifest[
        "forecast"
    ][
        "initialization_time"
    ] == "2026-07-24T00:00:00"

    assert manifest[
        "verification"
    ][
        "sid"
    ] == "2026204N08267"

    assert manifest[
        "verification"
    ][
        "minimum_overlap"
    ] == 3

    assert manifest[
        "ibtracs"
    ][
        "basin"
    ] == "EP"

    assert manifest[
        "plots"
    ][
        "field_lead_times_hours"
    ] == [
        54,
        78,
        96,
        120,
    ]

    assert (
        manifest["created_at"]
    )

def test_export_verification_manifest_without_plots(
    tmp_path,
):
    metadata = ForecastMetadata(
        model_name="graphcast",
        model_version="unknown",
        backend="earth2studio",
        initialization_time=datetime(
            2026,
            7,
            24,
        ),
    )

    path = export_verification_manifest(
        tmp_path,
        forecast_path="forecast.zarr",
        forecast_metadata=metadata,
        sid="2026204N08267",
        lat_min=5.0,
        lat_max=35.0,
        lon_min=-130.0,
        lon_max=-90.0,
        device="cpu",
        minimum_overlap=1,
        ibtracs_path="ibtracs.csv",
        ibtracs_basin="EP",
        generate_plots=False,
        field_lead_times=None,
    )

    with path.open(
        encoding="utf-8",
    ) as handle:
        manifest = json.load(
            handle
        )

    assert (
        manifest["plots"]["enabled"]
        is False
    )

    assert (
        manifest[
            "plots"
        ][
            "field_lead_times_hours"
        ]
        is None
    )