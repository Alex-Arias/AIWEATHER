import importlib.util
from pathlib import Path

import numpy as np
import xarray as xr


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "build_mexican_pacific_wave_guidance.py"
)

SPEC = importlib.util.spec_from_file_location(
    "build_mexican_pacific_wave_guidance",
    SCRIPT,
)

MODULE = importlib.util.module_from_spec(
    SPEC
)

SPEC.loader.exec_module(
    MODULE
)


def test_normalize_longitude():
    values = MODULE.normalize_longitude(
        np.array(
            [
                0.0,
                180.0,
                270.0,
                359.0,
            ]
        )
    )

    np.testing.assert_allclose(
        values,
        np.array(
            [
                0.0,
                -180.0,
                -90.0,
                -1.0,
            ]
        ),
    )


def test_reconstruct_cardinal_mwd():
    cases = [
        (0.0, 1.0, 0.0),
        (1.0, 0.0, 90.0),
        (0.0, -1.0, 180.0),
        (-1.0, 0.0, 270.0),
    ]

    for sinv, cosv, expected in cases:
        result = (
            MODULE.circular_mwd_from_components(
                np.array([sinv]),
                np.array([cosv]),
            )
        )

        assert np.isclose(
            result,
            expected,
        )


def test_circular_mean_wraparound():
    angles = np.array(
        [
            350.0,
            10.0,
        ]
    )

    sinv = np.sin(
        np.radians(
            angles
        )
    )

    cosv = np.cos(
        np.radians(
            angles
        )
    )

    result = (
        MODULE.circular_mwd_from_components(
            sinv,
            cosv,
        )
    )

    assert (
        np.isclose(
            result,
            0.0,
        )
        or np.isclose(
            result,
            360.0,
        )
    )


def test_propagation_conversion():
    expected = {
        0.0: 180.0,
        90.0: 270.0,
        180.0: 0.0,
        270.0: 90.0,
    }

    for source, target in expected.items():
        assert np.isclose(
            MODULE.propagation_from_mwd(
                source
            ),
            target,
        )


def synthetic_dataset():
    lat = np.array(
        [
            0.0,
            0.25,
        ]
    )

    lon = np.array(
        [
            359.75,
            0.0,
        ]
    )

    lead = np.array(
        [
            np.timedelta64(0, "h"),
            np.timedelta64(6, "h"),
        ]
    )

    shape = (
        1,
        2,
        2,
        2,
    )

    swh = np.full(
        shape,
        2.0,
    )

    swh[:, 1] = 3.0

    mwp = np.full(
        shape,
        10.0,
    )

    # ECMWF MWD-from = 270 deg.
    cos_mwd = np.full(
        shape,
        0.0,
    )

    sin_mwd = np.full(
        shape,
        -1.0,
    )

    return xr.Dataset(
        data_vars={
            "swh": (
                (
                    "time",
                    "lead_time",
                    "lat",
                    "lon",
                ),
                swh,
            ),

            "mwp": (
                (
                    "time",
                    "lead_time",
                    "lat",
                    "lon",
                ),
                mwp,
            ),

            "cos_mwd": (
                (
                    "time",
                    "lead_time",
                    "lat",
                    "lon",
                ),
                cos_mwd,
            ),

            "sin_mwd": (
                (
                    "time",
                    "lead_time",
                    "lat",
                    "lon",
                ),
                sin_mwd,
            ),
        },

        coords={
            "time":
                np.array(
                    [
                        np.datetime64(
                            "2026-09-28T12:00"
                        )
                    ]
                ),

            "lead_time":
                lead,

            "lat":
                lat,

            "lon":
                lon,
        },
    )


def test_build_guidance():
    ds = synthetic_dataset()

    anchors = (
        (
            "Test sector",
            "Test anchor",
            0.0,
            -0.1,
        ),
    )

    result = MODULE.build_guidance(
        ds,
        radius_km=100.0,
        anchors=anchors,
    )

    assert len(result) == 2

    assert (
        result["lead_time_hours"].tolist()
        == [0, 6]
    )

    assert (
        result["n_valid_cells"] > 0
    ).all()

    assert np.allclose(
        result["swh_mean_m"],
        [
            2.0,
            3.0,
        ],
    )

    assert np.allclose(
        result["mwp_mean_s"],
        10.0,
    )

    assert np.allclose(
        result["mwd_from_deg"],
        270.0,
    )

    assert np.allclose(
        result[
            "wave_propagation_deg"
        ],
        90.0,
    )

    assert (
        result["wave_data_available"]
    ).all()


def test_invalid_radius():
    ds = synthetic_dataset()

    try:
        MODULE.build_guidance(
            ds,
            radius_km=0.0,
            anchors=(),
        )
    except ValueError as exc:
        assert (
            "radius_km must be positive"
            in str(exc)
        )
    else:
        raise AssertionError(
            "Expected ValueError"
        )


def test_default_sampling_radius_is_100_km():
    assert np.isclose(
        MODULE.DEFAULT_RADIUS_KM,
        100.0,
    )


def test_sampling_support_requires_three_cells():
    ds = synthetic_dataset()

    # A small radius around a single grid cell should
    # retain wave data but fail the >=3-cell support test.
    anchors = (
        (
            "Test sector",
            "Test anchor",
            0.0,
            0.0,
        ),
    )

    result = MODULE.build_guidance(
        ds,
        radius_km=5.0,
        anchors=anchors,
    )

    assert (
        result["wave_data_available"]
    ).all()

    assert (
        result["n_valid_cells"] == 1
    ).all()

    assert (
        ~result["sampling_supported"]
    ).all()


def test_sampling_support_with_multiple_cells():
    ds = synthetic_dataset()

    anchors = (
        (
            "Test sector",
            "Test anchor",
            0.0,
            -0.1,
        ),
    )

    result = MODULE.build_guidance(
        ds,
        radius_km=100.0,
        anchors=anchors,
    )

    assert (
        result["n_valid_cells"] >= 3
    ).all()

    assert (
        result["sampling_supported"]
    ).all()
