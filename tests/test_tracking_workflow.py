import numpy as np

from aiweather.forecast import ForecastMetadata
from aiweather.tracking.genesis import GenesisResult
from aiweather.tracking.records import TrackRecord
from aiweather.tracking.workflow import (
    build_native_tc_track,
)


class DummyRegion:
    def __init__(self):
        self.data = {
            "lead_time": DummyArray(
                np.array(
                    [
                        np.timedelta64(24, "h"),
                        np.timedelta64(30, "h"),
                    ]
                )
            ),
            "msl": DummyField(),
            "u10m": DummyField(),
            "v10m": DummyField(),
        }

    def __getitem__(self, key):
        return self.data[key]


class DummyArray:
    def __init__(self, values):
        self.values = values


class DummyField:
    def isel(self, **kwargs):
        return self


class DummyForecast:
    def __init__(self):
        self.metadata = ForecastMetadata(
            model_name="graphcast",
            model_version="unknown",
            backend="earth2studio",
            forecast_id="test",
            initialization_time=np.datetime64(
                "2026-07-24T00:00:00"
            ).astype("datetime64[us]").astype(object),
        )

    def select_region(
        self,
        *,
        lat_min,
        lat_max,
        lon_min,
        lon_max,
    ):
        return DummyRegion()


def test_build_native_tc_track(
    monkeypatch,
):
    genesis = GenesisResult(
        track_index=0,
        genesis_lead_time_hours=24,
        latitude=10.0,
        longitude=250.0,
        pressure=100000.0,
        max_wind=20.0,
        qualifying_points=3,
    )

    expected_records = [
        TrackRecord(
            lead_time_hours=24,
            valid_time=np.datetime64(
                "2026-07-25T00:00:00"
            ),
            latitude=10.0,
            longitude=250.0,
            pressure=100000.0,
            pressure_units="Pa",
            max_wind=20.0,
            wind_units="m/s",
        )
    ]

    monkeypatch.setattr(
        "aiweather.tracking.workflow.detect_pressure_minima",
        lambda *args, **kwargs: ["candidate"],
    )

    monkeypatch.setattr(
        "aiweather.tracking.workflow.associate_candidates",
        lambda *args, **kwargs: ["track"],
    )

    monkeypatch.setattr(
        "aiweather.tracking.workflow.detect_genesis",
        lambda *args, **kwargs: [genesis],
    )

    monkeypatch.setattr(
        "aiweather.tracking.workflow.select_first_genesis",
        lambda *args, **kwargs: genesis,
    )

    monkeypatch.setattr(
        "aiweather.tracking.workflow.track_from_genesis",
        lambda *args, **kwargs: ["native-track"],
    )

    monkeypatch.setattr(
        "aiweather.tracking.workflow.build_track_records",
        lambda *args, **kwargs: expected_records,
    )

    result_genesis, records = build_native_tc_track(
        DummyForecast(),
        lat_min=5.0,
        lat_max=35.0,
        lon_min=-130.0,
        lon_max=-90.0,
        candidate_lead_min_hours=24,
        candidate_lead_max_hours=30,
    )

    assert result_genesis is genesis
    assert records is expected_records

import pytest


def test_build_native_tc_track_no_genesis(
    monkeypatch,
):
    monkeypatch.setattr(
        "aiweather.tracking.workflow.detect_pressure_minima",
        lambda *args, **kwargs: ["candidate"],
    )

    monkeypatch.setattr(
        "aiweather.tracking.workflow.associate_candidates",
        lambda *args, **kwargs: ["track"],
    )

    monkeypatch.setattr(
        "aiweather.tracking.workflow.detect_genesis",
        lambda *args, **kwargs: [],
    )

    monkeypatch.setattr(
        "aiweather.tracking.workflow.select_first_genesis",
        lambda *args, **kwargs: None,
    )

    with pytest.raises(
        RuntimeError,
        match="No tropical cyclone genesis",
    ):
        build_native_tc_track(
            DummyForecast(),
            lat_min=5.0,
            lat_max=35.0,
            lon_min=-130.0,
            lon_max=-90.0,
            candidate_lead_min_hours=24,
            candidate_lead_max_hours=30,
        )


def test_build_native_tc_track_missing_initialization_time(
    monkeypatch,
):
    genesis = GenesisResult(
        track_index=0,
        genesis_lead_time_hours=24,
        latitude=10.0,
        longitude=250.0,
        pressure=100000.0,
        max_wind=20.0,
        qualifying_points=3,
    )

    forecast = DummyForecast()
    forecast.metadata.initialization_time = None

    monkeypatch.setattr(
        "aiweather.tracking.workflow.detect_pressure_minima",
        lambda *args, **kwargs: ["candidate"],
    )

    monkeypatch.setattr(
        "aiweather.tracking.workflow.associate_candidates",
        lambda *args, **kwargs: ["track"],
    )

    monkeypatch.setattr(
        "aiweather.tracking.workflow.detect_genesis",
        lambda *args, **kwargs: [genesis],
    )

    monkeypatch.setattr(
        "aiweather.tracking.workflow.select_first_genesis",
        lambda *args, **kwargs: genesis,
    )

    monkeypatch.setattr(
        "aiweather.tracking.workflow.track_from_genesis",
        lambda *args, **kwargs: ["native-track"],
    )

    with pytest.raises(
        ValueError,
        match="initialization_time",
    ):
        build_native_tc_track(
            forecast,
            lat_min=5.0,
            lat_max=35.0,
            lon_min=-130.0,
            lon_max=-90.0,
            candidate_lead_min_hours=24,
            candidate_lead_max_hours=30,
        )

from aiweather.tracking.evaluation import TrackerEvaluation
from aiweather.tracking.workflow import (
    TrackingWorkflowResult,
    evaluate_forecast_trackers,
)


def test_evaluate_forecast_trackers(
    monkeypatch,
):
    genesis = GenesisResult(
        track_index=0,
        genesis_lead_time_hours=24,
        latitude=10.0,
        longitude=250.0,
        pressure=100000.0,
        max_wind=20.0,
        qualifying_points=3,
    )

    native_records = [
        TrackRecord(
            lead_time_hours=24,
            valid_time=np.datetime64(
                "2026-07-25T00:00:00"
            ),
            latitude=10.0,
            longitude=250.0,
            pressure=100000.0,
            pressure_units="Pa",
            max_wind=20.0,
            wind_units="m/s",
        )
    ]

    evaluation = TrackerEvaluation(
        reference_records=native_records,
        wuduan_match=None,
        vitart_match=None,
    )

    monkeypatch.setattr(
        "aiweather.tracking.workflow.build_native_tc_track",
        lambda *args, **kwargs: (
            genesis,
            native_records,
        ),
    )

    monkeypatch.setattr(
        "aiweather.tracking.earth2studio.run_wuduan_tracker",
        lambda *args, **kwargs: ["wuduan"],
    )

    monkeypatch.setattr(
        "aiweather.tracking.earth2studio.run_vitart_tracker",
        lambda *args, **kwargs: ["vitart"],
    )

    monkeypatch.setattr(
        "aiweather.tracking.evaluation.compare_tracker_ensemble",
        lambda *args, **kwargs: evaluation,
    )

    forecast = DummyForecast()
    forecast.dataset = object()

    result = evaluate_forecast_trackers(
        forecast,
        lat_min=5.0,
        lat_max=35.0,
        lon_min=-130.0,
        lon_max=-90.0,
    )

    assert isinstance(
        result,
        TrackingWorkflowResult,
    )

    assert result.genesis is genesis
    assert result.native_records is native_records
    assert result.evaluation is evaluation

def test_evaluate_forecast_trackers_native_only(
    monkeypatch,
):
    genesis = GenesisResult(
        track_index=0,
        genesis_lead_time_hours=24,
        latitude=10.0,
        longitude=250.0,
        pressure=100000.0,
        max_wind=20.0,
        qualifying_points=3,
    )

    native_records = [
        TrackRecord(
            lead_time_hours=24,
            valid_time=np.datetime64(
                "2026-07-25T00:00:00"
            ),
            latitude=10.0,
            longitude=250.0,
            pressure=100000.0,
            pressure_units="Pa",
            max_wind=20.0,
            wind_units="m/s",
        )
    ]

    evaluation = TrackerEvaluation(
        reference_records=native_records,
        wuduan_match=None,
        vitart_match=None,
    )

    monkeypatch.setattr(
        "aiweather.tracking.workflow.build_native_tc_track",
        lambda *args, **kwargs: (
            genesis,
            native_records,
        ),
    )

    monkeypatch.setattr(
        "aiweather.tracking.evaluation.compare_tracker_ensemble",
        lambda *args, **kwargs: evaluation,
    )

    forecast = DummyForecast()
    forecast.dataset = object()

    result = evaluate_forecast_trackers(
        forecast,
        lat_min=5.0,
        lat_max=35.0,
        lon_min=-130.0,
        lon_max=-90.0,
        run_wuduan=False,
        run_vitart=False,
    )

    assert result.genesis is genesis
    assert result.native_records is native_records
    assert result.evaluation is evaluation

def test_evaluate_forecast_trackers_external_reference(
    monkeypatch,
):
    genesis = GenesisResult(
        track_index=0,
        genesis_lead_time_hours=24,
        latitude=10.0,
        longitude=250.0,
        pressure=100000.0,
        max_wind=20.0,
        qualifying_points=3,
    )

    native_records = [
        TrackRecord(
            lead_time_hours=24,
            valid_time=np.datetime64(
                "2026-07-25T00:00:00"
            ),
            latitude=10.0,
            longitude=250.0,
            pressure=100000.0,
            pressure_units="Pa",
            max_wind=20.0,
            wind_units="m/s",
        )
    ]

    reference_records = [
        TrackRecord(
            lead_time_hours=24,
            valid_time=np.datetime64(
                "2026-07-25T00:00:00"
            ),
            latitude=15.0,
            longitude=245.0,
            pressure=99500.0,
            pressure_units="Pa",
            max_wind=25.0,
            wind_units="m/s",
        )
    ]

    monkeypatch.setattr(
        "aiweather.tracking.workflow.build_native_tc_track",
        lambda *args, **kwargs: (
            genesis,
            native_records,
        ),
    )

    monkeypatch.setattr(
        "aiweather.tracking.earth2studio.run_wuduan_tracker",
        lambda *args, **kwargs: ["wuduan"],
    )

    monkeypatch.setattr(
        "aiweather.tracking.earth2studio.run_vitart_tracker",
        lambda *args, **kwargs: ["vitart"],
    )

    captured = {}

    def fake_compare(
        reference,
        **kwargs,
    ):
        captured["reference"] = reference

        return TrackerEvaluation(
            reference_records=reference,
            wuduan_match=None,
            vitart_match=None,
        )

    monkeypatch.setattr(
        "aiweather.tracking.evaluation.compare_tracker_ensemble",
        fake_compare,
    )

    forecast = DummyForecast()
    forecast.dataset = object()

    result = evaluate_forecast_trackers(
        forecast,
        lat_min=5.0,
        lat_max=35.0,
        lon_min=-130.0,
        lon_max=-90.0,
        reference_records=reference_records,
    )

    assert captured["reference"] is reference_records
    assert result.native_records is native_records
    assert (
        result.evaluation.reference_records
        is reference_records
    )
