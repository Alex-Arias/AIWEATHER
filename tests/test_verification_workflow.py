import numpy as np
import pytest

from aiweather.tracking.evaluation import (
    TrackerEvaluation,
)
from aiweather.tracking.genesis import (
    GenesisResult,
)
from aiweather.tracking.records import (
    TrackRecord,
)
from aiweather.tracking.selection import (
    TrackMatch,
)
from aiweather.tracking.workflow import (
    TrackingWorkflowResult,
)
from aiweather.verification.comparison import (
    TrackVerification,
)
from aiweather.verification.workflow import (
    VerificationWorkflowResult,
    verify_tracking_workflow,
)


def make_record(
    lead_time_hours,
    valid_time,
    latitude,
    longitude,
):
    return TrackRecord(
        lead_time_hours=lead_time_hours,
        valid_time=np.datetime64(
            valid_time
        ),
        latitude=latitude,
        longitude=longitude,
        pressure=100000.0,
        pressure_units="Pa",
        max_wind=20.0,
        wind_units="m/s",
    )


def make_verification(
    name,
):
    return TrackVerification(
        forecast_name=name,
        observation_name="ibtracs",
        table=None,
    )


def test_verify_tracking_workflow_all_trackers(
    monkeypatch,
):
    native_records = [
        make_record(
            54,
            "2026-07-26T06:00:00",
            13.0,
            254.0,
        )
    ]

    wuduan_records = [
        make_record(
            54,
            "2026-07-26T06:00:00",
            13.0,
            254.0,
        )
    ]

    vitart_records = [
        make_record(
            96,
            "2026-07-28T00:00:00",
            16.75,
            248.75,
        )
    ]

    observations = [
        make_record(
            54,
            "2026-07-26T06:00:00",
            12.6,
            -107.6,
        )
    ]

    dummy_comparison = TrackVerification(
        forecast_name="dummy",
        observation_name="ibtracs",
        table=None,
    )

    wuduan_match = type(
        "DummyMatch",
        (),
        {
            "records": wuduan_records,
        },
    )()

    vitart_match = type(
        "DummyMatch",
        (),
        {
            "records": vitart_records,
        },
    )()

    evaluation = TrackerEvaluation(
        reference_records=native_records,
        wuduan_match=wuduan_match,
        vitart_match=vitart_match,
    )

    tracking_result = TrackingWorkflowResult(
        genesis=GenesisResult(
            track_index=0,
            genesis_lead_time_hours=54,
            latitude=13.0,
            longitude=254.0,
            pressure=100000.0,
            max_wind=20.0,
            qualifying_points=3,
        ),
        native_records=native_records,
        evaluation=evaluation,
    )

    calls = []

    def fake_compare(
        forecast_records,
        observation_records,
        *,
        forecast_name,
        observation_name,
    ):
        calls.append(
            (
                forecast_name,
                forecast_records,
                observation_records,
                observation_name,
            )
        )

        return TrackVerification(
            forecast_name=forecast_name,
            observation_name=observation_name,
            table=None,
        )

    monkeypatch.setattr(
        "aiweather.verification.workflow."
        "compare_forecast_to_best_track",
        fake_compare,
    )

    result = verify_tracking_workflow(
        tracking_result,
        observations,
        observation_name="ibtracs",
    )

    assert isinstance(
        result,
        VerificationWorkflowResult,
    )

    assert result.native is not None
    assert result.wuduan is not None
    assert result.vitart is not None

    assert [
        item[0]
        for item in calls
    ] == [
        "native",
        "wuduan",
        "vitart",
    ]

    assert result.observations is observations


def test_verify_tracking_workflow_native_only(
    monkeypatch,
):
    native_records = [
        make_record(
            54,
            "2026-07-26T06:00:00",
            13.0,
            254.0,
        )
    ]

    observations = [
        make_record(
            54,
            "2026-07-26T06:00:00",
            12.6,
            -107.6,
        )
    ]

    tracking_result = TrackingWorkflowResult(
        genesis=GenesisResult(
            track_index=0,
            genesis_lead_time_hours=54,
            latitude=13.0,
            longitude=254.0,
            pressure=100000.0,
            max_wind=20.0,
            qualifying_points=3,
        ),
        native_records=native_records,
        evaluation=None,
    )

    monkeypatch.setattr(
        "aiweather.verification.workflow."
        "compare_forecast_to_best_track",
        lambda *args, **kwargs: TrackVerification(
            forecast_name="native",
            observation_name="ibtracs",
            table=None,
        ),
    )

    result = verify_tracking_workflow(
        tracking_result,
        observations,
        observation_name="ibtracs",
    )

    assert result.native is not None
    assert result.wuduan is None
    assert result.vitart is None

    assert set(
        result.verifications
    ) == {
        "native",
    }


def test_verifications_property_all_trackers(
    monkeypatch,
):
    native_records = [
        make_record(
            54,
            "2026-07-26T06:00:00",
            13.0,
            254.0,
        )
    ]

    observations = [
        make_record(
            54,
            "2026-07-26T06:00:00",
            12.6,
            -107.6,
        )
    ]

    evaluation = TrackerEvaluation(
        reference_records=native_records,
        wuduan_match=type(
            "DummyMatch",
            (),
            {
                "records": native_records,
            },
        )(),
        vitart_match=type(
            "DummyMatch",
            (),
            {
                "records": native_records,
            },
        )(),
    )

    tracking_result = TrackingWorkflowResult(
        genesis=GenesisResult(
            track_index=0,
            genesis_lead_time_hours=54,
            latitude=13.0,
            longitude=254.0,
            pressure=100000.0,
            max_wind=20.0,
            qualifying_points=3,
        ),
        native_records=native_records,
        evaluation=evaluation,
    )

    monkeypatch.setattr(
        "aiweather.verification.workflow."
        "compare_forecast_to_best_track",
        lambda *args, forecast_name, **kwargs:
            TrackVerification(
                forecast_name=forecast_name,
                observation_name="ibtracs",
                table=None,
            ),
    )

    result = verify_tracking_workflow(
        tracking_result,
        observations,
    )

    assert set(
        result.verifications
    ) == {
        "native",
        "wuduan",
        "vitart",
    }


def test_verify_tracking_workflow_invalid_tracking_result():
    with pytest.raises(
        TypeError,
        match="TrackingWorkflowResult",
    ):
        verify_tracking_workflow(
            "invalid",
            [],
        )


def test_verify_tracking_workflow_invalid_observations():
    native_records = []

    tracking_result = TrackingWorkflowResult(
        genesis=GenesisResult(
            track_index=0,
            genesis_lead_time_hours=54,
            latitude=13.0,
            longitude=254.0,
            pressure=100000.0,
            max_wind=20.0,
            qualifying_points=3,
        ),
        native_records=native_records,
        evaluation=None,
    )

    with pytest.raises(
        TypeError,
        match="observation_records",
    ):
        verify_tracking_workflow(
            tracking_result,
            "invalid",
        )