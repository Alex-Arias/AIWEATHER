import importlib.util
from pathlib import Path

import numpy as np

from aiweather.tracking.earth2studio import Earth2StudioTrack


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "export_multisystem_tracker_tracks.py"
)

SPEC = importlib.util.spec_from_file_location(
    "export_multisystem_tracker_tracks",
    SCRIPT_PATH,
)

MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)

select_originating_regional_track = (
    MODULE.select_originating_regional_track
)


def make_track(
    path_id,
    first_lead,
    last_lead,
    n,
    latitude=15.0,
    longitude=232.0,
):
    lead_time_hours = np.linspace(
        first_lead,
        last_lead,
        n,
        dtype=int,
    )

    return Earth2StudioTrack(
        path_id=path_id,
        lead_time_hours=lead_time_hours,
        latitude=np.full(n, latitude),
        longitude=np.full(n, longitude),
        pressure=np.full(n, 100000.0),
        max_wind=np.full(n, 20.0),
    )


def select(tracks):
    return select_originating_regional_track(
        tracks,
        lat_min=5.0,
        lat_max=35.0,
        lon_min=220.0,
        lon_max=245.0,
    )


def test_prefers_earlier_origin_over_longer_duration():
    # Reproduces the Pangu3/Odalys Cycle-4 ambiguity:
    #
    # path 3:
    #     starts at lead 0 h
    #     duration 132 h
    #     45 tracker points
    #
    # path 22:
    #     starts at lead 96 h
    #     duration 141 h
    #     44 tracker points
    #
    # The operational selector must retain the path that
    # represents the system from initialization rather than
    # switching to a later-originating path solely because
    # that later path has slightly greater temporal support.

    early = make_track(
        path_id=3,
        first_lead=0,
        last_lead=132,
        n=45,
    )

    later = make_track(
        path_id=22,
        first_lead=96,
        last_lead=237,
        n=44,
    )

    selected = select([later, early])

    assert selected is early
    assert selected.path_id == 3


def test_same_origin_prefers_longer_duration():
    shorter = make_track(
        path_id=10,
        first_lead=0,
        last_lead=120,
        n=21,
    )

    longer = make_track(
        path_id=11,
        first_lead=0,
        last_lead=132,
        n=20,
    )

    selected = select([shorter, longer])

    assert selected is longer
    assert selected.path_id == 11


def test_same_origin_and_duration_prefers_more_points():
    fewer = make_track(
        path_id=20,
        first_lead=0,
        last_lead=132,
        n=20,
    )

    more = make_track(
        path_id=21,
        first_lead=0,
        last_lead=132,
        n=23,
    )

    selected = select([fewer, more])

    assert selected is more
    assert selected.path_id == 21


def test_final_tie_breaker_prefers_lower_path_id():
    high_id = make_track(
        path_id=31,
        first_lead=0,
        last_lead=132,
        n=23,
    )

    low_id = make_track(
        path_id=30,
        first_lead=0,
        last_lead=132,
        n=23,
    )

    selected = select([high_id, low_id])

    assert selected is low_id
    assert selected.path_id == 30


def test_rejects_short_detection():
    short = make_track(
        path_id=40,
        first_lead=0,
        last_lead=9,
        n=4,
    )

    assert select([short]) is None


def test_rejects_track_originating_outside_region():
    outside = make_track(
        path_id=50,
        first_lead=0,
        last_lead=132,
        n=23,
        longitude=250.0,
    )

    assert select([outside]) is None
