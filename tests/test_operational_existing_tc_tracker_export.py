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
