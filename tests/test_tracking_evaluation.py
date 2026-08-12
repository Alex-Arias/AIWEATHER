import numpy as np
import pytest

from aiweather.tracking.earth2studio import (
    Earth2StudioTrack,
)
from aiweather.tracking.evaluation import (
    TrackerEvaluation,
    compare_tracker_ensemble,
)
from aiweather.tracking.records import (
    TrackRecord,
)


INITIALIZATION_TIME = np.datetime64(
    "2026-07-24T00:00:00"
)


def make_record(
    lead_time_hours,
    latitude,
    longitude,
):
    return TrackRecord(
        lead_time_hours=lead_time_hours,
        valid_time=(
            INITIALIZATION_TIME
            + np.timedelta64(
                lead_time_hours,
                "h",
            )
        ),
        latitude=latitude,
        longitude=longitude,
        pressure=100000.0,
        pressure_units="Pa",
        max_wind=20.0,
        wind_units="m/s",
    )


def make_track(
    path_id,
    lead_times,
    latitudes,
    longitudes,
):
    count = len(lead_times)

    return Earth2StudioTrack(
        path_id=path_id,
        lead_time_hours=np.asarray(
            lead_times,
            dtype=int,
        ),
        latitude=np.asarray(
            latitudes,
            dtype=float,
        ),
        longitude=np.asarray(
            longitudes,
            dtype=float,
        ),
        pressure=np.full(
            count,
            100000.0,
            dtype=float,
        ),
        max_wind=np.full(
            count,
            20.0,
            dtype=float,
        ),
    )


def test_compare_tracker_ensemble_both_trackers():
    reference = [
        make_record(0, 10.0, 250.0),
        make_record(6, 11.0, 249.0),
        make_record(12, 12.0, 248.0),
    ]

    wuduan_tracks = [
        make_track(
            1,
            [0, 6, 12],
            [30.0, 31.0, 32.0],
            [200.0, 199.0, 198.0],
        ),
        make_track(
            8,
            [0, 6, 12],
            [10.0, 11.0, 12.0],
            [250.0, 249.0, 248.0],
        ),
    ]

    vitart_tracks = [
        make_track(
            0,
            [0, 6, 12],
            [40.0, 41.0, 42.0],
            [100.0, 99.0, 98.0],
        ),
        make_track(
            2,
            [0, 6, 12],
            [10.0, 11.0, 12.0],
            [250.0, 249.0, 248.0],
        ),
    ]

    evaluation = compare_tracker_ensemble(
        reference,
        initialization_time=INITIALIZATION_TIME,
        wuduan_tracks=wuduan_tracks,
        vitart_tracks=vitart_tracks,
        minimum_overlap=2,
    )

    assert isinstance(
        evaluation,
        TrackerEvaluation,
    )

    assert evaluation.wuduan_match is not None
    assert evaluation.vitart_match is not None

    assert evaluation.wuduan_match.path_id == 8
    assert evaluation.vitart_match.path_id == 2

    assert (
        evaluation.wuduan_match.mean_track_error_km
        == pytest.approx(0.0)
    )

    assert (
        evaluation.vitart_match.mean_track_error_km
        == pytest.approx(0.0)
    )


def test_compare_tracker_ensemble_wuduan_only():
    reference = [
        make_record(0, 10.0, 250.0),
    ]

    wuduan_tracks = [
        make_track(
            8,
            [0],
            [10.0],
            [250.0],
        ),
    ]

    evaluation = compare_tracker_ensemble(
        reference,
        initialization_time=INITIALIZATION_TIME,
        wuduan_tracks=wuduan_tracks,
    )

    assert evaluation.wuduan_match is not None
    assert evaluation.wuduan_match.path_id == 8
    assert evaluation.vitart_match is None


def test_compare_tracker_ensemble_vitart_only():
    reference = [
        make_record(0, 10.0, 250.0),
    ]

    vitart_tracks = [
        make_track(
            2,
            [0],
            [10.0],
            [250.0],
        ),
    ]

    evaluation = compare_tracker_ensemble(
        reference,
        initialization_time=INITIALIZATION_TIME,
        vitart_tracks=vitart_tracks,
    )

    assert evaluation.wuduan_match is None
    assert evaluation.vitart_match is not None
    assert evaluation.vitart_match.path_id == 2


def test_compare_tracker_ensemble_no_candidates():
    reference = [
        make_record(0, 10.0, 250.0),
    ]

    evaluation = compare_tracker_ensemble(
        reference,
        initialization_time=INITIALIZATION_TIME,
    )

    assert evaluation.wuduan_match is None
    assert evaluation.vitart_match is None
    assert evaluation.comparisons == {}


def test_tracker_evaluation_comparisons_property():
    reference = [
        make_record(0, 10.0, 250.0),
    ]

    wuduan_tracks = [
        make_track(
            8,
            [0],
            [10.0],
            [250.0],
        ),
    ]

    vitart_tracks = [
        make_track(
            2,
            [0],
            [10.0],
            [250.0],
        ),
    ]

    evaluation = compare_tracker_ensemble(
        reference,
        initialization_time=INITIALIZATION_TIME,
        wuduan_tracks=wuduan_tracks,
        vitart_tracks=vitart_tracks,
    )

    comparisons = evaluation.comparisons

    assert set(comparisons) == {
        "wuduan",
        "vitart",
    }

    assert (
        comparisons["wuduan"]
        is evaluation.wuduan_match.comparison
    )

    assert (
        comparisons["vitart"]
        is evaluation.vitart_match.comparison
    )


def test_compare_tracker_ensemble_minimum_overlap():
    reference = [
        make_record(0, 10.0, 250.0),
        make_record(6, 11.0, 249.0),
    ]

    wuduan_tracks = [
        make_track(
            8,
            [6, 12],
            [11.0, 12.0],
            [249.0, 248.0],
        ),
    ]

    evaluation = compare_tracker_ensemble(
        reference,
        initialization_time=INITIALIZATION_TIME,
        wuduan_tracks=wuduan_tracks,
        minimum_overlap=2,
    )

    assert evaluation.wuduan_match is None


def test_compare_tracker_ensemble_rejects_invalid_overlap():
    with pytest.raises(
        ValueError,
        match="minimum_overlap",
    ):
        compare_tracker_ensemble(
            [],
            initialization_time=INITIALIZATION_TIME,
            minimum_overlap=0,
        )


def test_compare_tracker_ensemble_rejects_invalid_reference():
    with pytest.raises(
        TypeError,
        match="reference_records",
    ):
        compare_tracker_ensemble(
            "invalid",
            initialization_time=INITIALIZATION_TIME,
        )
