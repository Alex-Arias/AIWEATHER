from datetime import datetime
import importlib.util
from pathlib import Path

import numpy as np

from aiweather.tracking import (
    TrackRecord,
)
from aiweather.tracking.earth2studio import Earth2StudioTrack


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "export_existing_tc_tracker_tracks.py"
)

SPEC = importlib.util.spec_from_file_location(
    "export_existing_tc_tracker_tracks",
    SCRIPT_PATH,
)

MODULE = importlib.util.module_from_spec(SPEC)

if SCRIPT_PATH.exists():
    SPEC.loader.exec_module(MODULE)


def make_candidate(
    path_id,
    latitude,
    longitude,
):
    leads = np.arange(
        0,
        49,
        6,
        dtype=int,
    )

    return Earth2StudioTrack(
        path_id=path_id,
        lead_time_hours=leads,
        latitude=np.full(
            len(leads),
            latitude,
        ),
        longitude=np.full(
            len(leads),
            longitude,
        ),
        pressure=np.full(
            len(leads),
            100000.0,
        ),
        max_wind=np.full(
            len(leads),
            20.0,
        ),
    )


def make_rachel_reference():
    records = []

    initialization = np.datetime64(
        "2026-09-28T12:00:00"
    )

    for lead in range(0, 49, 6):
        records.append(
            TrackRecord(
                lead_time_hours=lead,
                valid_time=(
                    initialization
                    + np.timedelta64(
                        lead,
                        "h",
                    )
                ),
                latitude=(
                    13.3
                    + 0.15 * (lead / 6)
                ),
                longitude=(
                    259.1
                    - 0.25 * (lead / 6)
                ),
                pressure=100000.0,
                max_wind=20.0,
            )
        )

    return records


def test_cycle1_matches_tracker_to_seeded_native_reference():
    assert SCRIPT_PATH.exists()

    reference = make_rachel_reference()

    # Wrong EPAC candidate: a different cyclone well northwest
    # of Rachel, but still inside a broad operational domain.
    other_system = make_candidate(
        path_id=10,
        latitude=23.5,
        longitude=246.0,
    )

    # Rachel-like candidate near the seeded Native reference.
    rachel = make_candidate(
        path_id=11,
        latitude=13.8,
        longitude=258.5,
    )

    match = MODULE.select_existing_storm_track(
        reference,
        [other_system, rachel],
        initialization_time="2026-09-28T12:00:00",
        minimum_overlap=2,
        maximum_mean_error_km=600.0,
    )

    assert match is not None
    assert match.track is rachel
    assert match.track.path_id == 11
    assert match.overlap_count >= 2
    assert match.mean_track_error_km < 600.0


def make_candidate_track(
    path_id,
    leads,
    latitudes,
    longitudes,
):
    """Build an Earth2Studio candidate with explicit trajectory."""
    leads = np.asarray(leads, dtype=int)

    return Earth2StudioTrack(
        path_id=path_id,
        lead_time_hours=leads,
        latitude=np.asarray(latitudes, dtype=float),
        longitude=np.asarray(longitudes, dtype=float),
        pressure=np.full(
            len(leads),
            100000.0,
            dtype=float,
        ),
        max_wind=np.full(
            len(leads),
            20.0,
            dtype=float,
        ),
    )


def make_reference_records(
    initialization,
    leads,
    latitudes,
    longitudes,
):
    """Build TrackRecord reference data at explicit valid times."""
    initialization = np.datetime64(initialization)

    records = []

    for lead, lat, lon in zip(
        leads,
        latitudes,
        longitudes,
    ):
        records.append(
            TrackRecord(
                lead_time_hours=int(lead),
                valid_time=(
                    initialization
                    + np.timedelta64(
                        int(lead),
                        "h",
                    )
                ),
                latitude=float(lat),
                longitude=float(lon),
                pressure=100000.0,
                max_wind=20.0,
            )
        )

    return records


def test_cycle2_continuity_uses_early_valid_time_window():
    """
    A legitimate current-cycle continuation must not be rejected
    because consecutive forecasts diverge late in the forecast.
    """

    previous_init = "2026-09-28T12:00:00"
    current_init = "2026-09-29T12:00:00"

    # Previous-cycle tracker:
    # valid times shared with the current cycle begin at previous +24 h.
    previous = make_reference_records(
        previous_init,
        [24, 30, 36, 42, 48, 54, 60, 66, 72,
         78, 84, 90, 96, 102, 108, 114, 120],
        [14.5, 14.7, 14.9, 15.1, 15.3, 15.5, 15.7, 15.9, 16.1,
         16.3, 16.5, 16.7, 16.9, 17.1, 17.3, 17.5, 17.7],
        [256.3, 256.0, 255.7, 255.4, 255.1, 254.8, 254.5, 254.2, 253.9,
         253.6, 253.3, 253.0, 252.7, 252.4, 252.1, 251.8, 251.5],
    )

    leads = np.arange(0, 121, 6)

    # Same storm: close during the early overlapping valid times,
    # but deliberately diverges strongly after current +72 h.
    latitudes = []
    longitudes = []

    for lead in leads:
        latitudes.append(
            14.5 + 0.2 * (lead / 6)
        )

        if lead <= 72:
            longitudes.append(
                256.3 - 0.3 * (lead / 6)
            )
        else:
            longitudes.append(
                245.0 - 1.5 * ((lead - 72) / 6)
            )

    candidate = make_candidate_track(
        4,
        leads,
        latitudes,
        longitudes,
    )

    match = MODULE.select_existing_storm_track_continuity(
        previous,
        [candidate],
        initialization_time=current_init,
        minimum_overlap=2,
        maximum_mean_error_km=600.0,
        continuity_window_hours=72,
    )

    assert match is not None
    assert match.path_id == 4


def test_cycle2_continuity_rejects_wrong_storm_early():
    """
    A different cyclone must still be rejected when it is already
    spatially inconsistent during the early continuity window.
    """

    previous = make_reference_records(
        "2026-09-28T12:00:00",
        [24, 30, 36, 42, 48, 54],
        [14.5, 14.7, 14.9, 15.1, 15.3, 15.5],
        [256.3, 256.0, 255.7, 255.4, 255.1, 254.8],
    )

    wrong = make_candidate_track(
        10,
        [0, 6, 12, 18, 24, 30],
        [24.0, 24.2, 24.4, 24.6, 24.8, 25.0],
        [245.0, 244.8, 244.6, 244.4, 244.2, 244.0],
    )

    match = MODULE.select_existing_storm_track_continuity(
        previous,
        [wrong],
        initialization_time="2026-09-29T12:00:00",
        minimum_overlap=2,
        maximum_mean_error_km=600.0,
        continuity_window_hours=72,
    )

    assert match is None


def test_cycle2_continuity_prefers_correct_candidate():
    """
    When multiple systems exist, continuity should select the
    candidate closest to the previous-cycle Rachel trajectory.
    """

    previous = make_reference_records(
        "2026-09-28T12:00:00",
        [24, 30, 36, 42, 48, 54],
        [14.5, 14.7, 14.9, 15.1, 15.3, 15.5],
        [256.3, 256.0, 255.7, 255.4, 255.1, 254.8],
    )

    rachel = make_candidate_track(
        4,
        [0, 6, 12, 18, 24, 30],
        [14.6, 14.8, 15.0, 15.2, 15.4, 15.6],
        [256.2, 255.9, 255.6, 255.3, 255.0, 254.7],
    )

    other = make_candidate_track(
        9,
        [0, 6, 12, 18, 24, 30],
        [21.0, 21.2, 21.4, 21.6, 21.8, 22.0],
        [247.0, 246.8, 246.6, 246.4, 246.2, 246.0],
    )

    match = MODULE.select_existing_storm_track_continuity(
        previous,
        [other, rachel],
        initialization_time="2026-09-29T12:00:00",
        minimum_overlap=2,
        maximum_mean_error_km=600.0,
        continuity_window_hours=72,
    )

    assert match is not None
    assert match.path_id == 4


def test_cycle2_continuity_rebases_previous_leads_to_current_init(
    monkeypatch,
):
    """
    Previous-cycle valid times must be expressed relative to the
    current initialization before lead-time track matching.

    For daily 12Z cycles:
        previous +24 h -> current +0 h
        previous +30 h -> current +6 h
        previous +36 h -> current +12 h
    """
    previous = make_reference_records(
        "2026-09-28T12:00:00",
        [24, 30, 36],
        [14.5, 14.7, 14.9],
        [256.3, 256.0, 255.7],
    )

    candidate = make_candidate_track(
        4,
        [0, 6, 12],
        [14.5, 14.7, 14.9],
        [256.3, 256.0, 255.7],
    )

    captured = {}

    real_select = MODULE.select_matching_track

    def capture_select(
        reference_records,
        candidate_tracks,
        **kwargs,
    ):
        captured["reference_records"] = reference_records

        return real_select(
            reference_records,
            candidate_tracks,
            **kwargs,
        )

    monkeypatch.setattr(
        MODULE,
        "select_matching_track",
        capture_select,
    )

    match = MODULE.select_existing_storm_track_continuity(
        previous,
        [candidate],
        initialization_time="2026-09-29T12:00:00",
        minimum_overlap=2,
        maximum_mean_error_km=600.0,
        continuity_window_hours=72,
    )

    assert match is not None
    assert match.path_id == 4

    rebased = captured["reference_records"]

    assert [
        record.lead_time_hours
        for record in rebased
    ] == [0, 6, 12]

    assert [
        datetime.fromisoformat(
            str(record.valid_time)
        )
        for record in rebased
    ] == [
        datetime(2026, 9, 29, 12),
        datetime(2026, 9, 29, 18),
        datetime(2026, 9, 30, 0),
    ]
