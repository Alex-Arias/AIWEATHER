"""Tests for scripts/tc_track_variability.py."""

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Load the standalone script as a module without requiring scripts/__init__.py.
# ---------------------------------------------------------------------------

SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "tc_track_variability.py"
)

SPEC = spec_from_file_location(
    "tc_track_variability",
    SCRIPT,
)

tcv = module_from_spec(SPEC)
SPEC.loader.exec_module(tcv)


# ---------------------------------------------------------------------------
# Spherical geometry
# ---------------------------------------------------------------------------

def test_spherical_mean_identical_points():
    lat, lon = tcv.spherical_mean(
        [20.0, 20.0, 20.0],
        [-110.0, -110.0, -110.0],
    )

    assert lat == pytest.approx(20.0)
    assert lon == pytest.approx(-110.0)


def test_spherical_mean_handles_dateline():
    lat, lon = tcv.spherical_mean(
        [10.0, 10.0],
        [179.0, -179.0],
    )

    assert lat == pytest.approx(
        10.001492526984078,
        abs=1e-10,
    )

    # Mean must remain near the dateline rather than incorrectly
    # averaging to 0 degrees longitude.
    assert abs(abs(lon) - 180.0) < 1e-10


def test_haversine_zero_distance():
    distance = tcv.haversine_km(
        20.0,
        -110.0,
        np.array([20.0]),
        np.array([-110.0]),
    )

    assert distance.shape == (1,)
    assert distance[0] == pytest.approx(0.0)


def test_haversine_one_degree_equator():
    distance = tcv.haversine_km(
        0.0,
        0.0,
        np.array([0.0]),
        np.array([1.0]),
    )

    # One degree of longitude at the equator is approximately
    # 111.195 km for the mean Earth radius used by the script.
    assert distance[0] == pytest.approx(
        111.195,
        abs=0.01,
    )


# ---------------------------------------------------------------------------
# Variability calculation
# ---------------------------------------------------------------------------

def synthetic_members():
    """Return a small deterministic multimodel/multicycle dataset."""

    rows = []

    valid_times = pd.to_datetime(
        [
            "2026-09-26 12:00:00",
            "2026-09-26 18:00:00",
        ]
    )

    # Four symmetric members around a nominal center.
    offsets = [
        ("aifs2", 0.2, 0.0),
        ("graphcast", -0.2, 0.0),
        ("pangu3", 0.0, 0.2),
        ("pangu6", 0.0, -0.2),
    ]

    for cycle in [1, 2, 3]:

        for valid_time in valid_times:

            for model, dlat, dlon in offsets:

                rows.append(
                    {
                        "valid_time": valid_time,
                        "cycle": cycle,
                        "model": model,
                        "latitude": 20.0 + dlat,
                        "longitude": -110.0 + dlon,
                    }
                )

    return pd.DataFrame(rows)


def test_compute_variability_counts_members_cycles_models():
    members = synthetic_members()

    result = tcv.compute_variability(members)

    assert len(result) == 2

    assert result["n_members"].tolist() == [12, 12]
    assert result["n_cycles"].tolist() == [3, 3]
    assert result["n_models"].tolist() == [4, 4]


def test_compute_variability_consensus_near_center():
    members = synthetic_members()

    result = tcv.compute_variability(members)

    np.testing.assert_allclose(
        result["consensus_latitude"],
        20.000112,
        atol=2e-4,
    )

    np.testing.assert_allclose(
        result["consensus_longitude"],
        -110.0,
        atol=1e-10,
    )


def test_compute_variability_spread_is_positive_and_ordered():
    members = synthetic_members()

    result = tcv.compute_variability(members)

    assert (result["r67_km"] > 0.0).all()
    assert (result["r90_km"] > 0.0).all()

    assert (
        result["r67_km"]
        <= result["r90_km"]
    ).all()

    assert (
        result["r90_km"]
        <= result["max_distance_km"]
    ).all()


def test_compute_variability_identical_members_zero_spread():
    members = pd.DataFrame(
        {
            "valid_time": pd.to_datetime(
                ["2026-09-26 12:00:00"] * 4
            ),
            "cycle": [1, 1, 1, 1],
            "model": [
                "aifs2",
                "graphcast",
                "pangu3",
                "pangu6",
            ],
            "latitude": [20.0] * 4,
            "longitude": [-110.0] * 4,
        }
    )

    result = tcv.compute_variability(members)

    row = result.iloc[0]

    assert row["n_members"] == 4
    assert row["n_cycles"] == 1
    assert row["n_models"] == 4

    assert row["r67_km"] == pytest.approx(0.0)
    assert row["r90_km"] == pytest.approx(0.0)
    assert row["max_distance_km"] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Plot-support geometry
# ---------------------------------------------------------------------------

def test_destination_point_zero_distance():
    lat, lon = tcv.destination_point(
        20.0,
        -110.0,
        45.0,
        0.0,
    )

    assert lat == pytest.approx(20.0)
    assert lon == pytest.approx(-110.0)


def test_initial_bearing_due_north():
    bearing = tcv.initial_bearing(
        20.0,
        -110.0,
        21.0,
        -110.0,
    )

    assert bearing == pytest.approx(0.0)


def test_automatic_extent_contains_all_members():
    members = synthetic_members()

    extent = tcv.automatic_extent(members)

    lon_min, lon_max, lat_min, lat_max = extent

    assert lon_min < members["longitude"].min()
    assert lon_max > members["longitude"].max()

    assert lat_min < members["latitude"].min()
    assert lat_max > members["latitude"].max()


# ---------------------------------------------------------------------------
# Longitude normalization
# ---------------------------------------------------------------------------

def test_normalize_longitude_converts_0_360_western_longitudes():
    lon = np.array([
        240.0,
        223.86763643605653,
    ])

    result = tcv.normalize_longitude(lon)

    np.testing.assert_allclose(
        result,
        [
            -120.0,
            -136.13236356394347,
        ],
        atol=1.0e-10,
    )


def test_normalize_longitude_preserves_negative_western_longitudes():
    lon = np.array([
        -120.0,
        -136.1323635639435,
    ])

    result = tcv.normalize_longitude(lon)

    np.testing.assert_allclose(
        result,
        lon,
        atol=1.0e-10,
    )


def test_normalize_longitude_boundary_convention():
    lon = np.array([
        0.0,
        180.0,
        360.0,
        -180.0,
    ])

    result = tcv.normalize_longitude(lon)

    np.testing.assert_allclose(
        result,
        [
            0.0,
            -180.0,
            0.0,
            -180.0,
        ],
        atol=1.0e-10,
    )
