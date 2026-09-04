import pandas as pd
import pytest

from scripts.wave_centers import (
    load_wave_centers,
    longitude_near_reference,
)


def test_load_operational_wave_centers(tmp_path):
    path = tmp_path / "operational_track.csv"

    pd.DataFrame(
        {
            "lead_time_hours": [12, 0, 6, 6],
            "latitude": [17.0, 16.0, 16.5, 99.0],
            "longitude": [241.0, 242.5, 241.75, 100.0],
        }
    ).to_csv(
        path,
        index=False,
    )

    case = {
        "case_id": "karina_test",
        "center_source": "operational",
        "center_path": str(path),
    }

    centers = load_wave_centers(case)

    assert centers["lead_time_hours"].tolist() == [
        0,
        6,
        12,
    ]

    assert centers["center_latitude"].tolist() == [
        16.0,
        16.5,
        17.0,
    ]

    assert centers["center_longitude"].tolist() == [
        242.5,
        241.75,
        241.0,
    ]

    assert centers["center_longitude_plot"].tolist() == pytest.approx(
        [
            -117.5,
            -118.25,
            -119.0,
        ]
    )


def test_operational_wave_centers_require_path():
    case = {
        "case_id": "karina_test",
        "center_source": "operational",
    }

    with pytest.raises(
        ValueError,
        match="has no center_path",
    ):
        load_wave_centers(case)


def test_operational_wave_centers_require_columns(tmp_path):
    path = tmp_path / "bad_track.csv"

    pd.DataFrame(
        {
            "lead_time_hours": [0],
            "latitude": [16.0],
        }
    ).to_csv(
        path,
        index=False,
    )

    case = {
        "case_id": "karina_test",
        "center_source": "operational",
        "center_path": str(path),
    }

    with pytest.raises(
        ValueError,
        match="missing columns",
    ):
        load_wave_centers(case)


def test_reject_unsupported_wave_center_source():
    case = {
        "case_id": "test_case",
        "center_source": "unknown",
    }

    with pytest.raises(
        ValueError,
        match="Unsupported wave center source",
    ):
        load_wave_centers(case)

def test_longitude_near_reference_crosses_dateline():
    result = longitude_near_reference(
        [
            -154.7,
            -175.0,
            179.0,
            170.0,
        ],
        -154.7,
    )

    assert result.tolist() == pytest.approx(
        [
            -154.7,
            -175.0,
            -181.0,
            -190.0,
        ]
    )


def test_operational_wave_centers_unwrap_dateline(tmp_path):
    path = tmp_path / "operational_track.csv"

    pd.DataFrame(
        {
            "lead_time_hours": [
                0,
                6,
                12,
                18,
            ],
            "latitude": [
                13.3,
                14.0,
                15.0,
                16.0,
            ],
            "longitude": [
                205.3,
                185.0,
                179.0,
                170.0,
            ],
        }
    ).to_csv(
        path,
        index=False,
    )

    case = {
        "case_id": "lowell_test",
        "center_source": "operational",
        "center_path": str(path),
        "longitude_reference": -154.7,
    }

    centers = load_wave_centers(
        case
    )

    assert centers[
        "center_longitude_plot"
    ].tolist() == pytest.approx(
        [
            -154.7,
            -175.0,
            -181.0,
            -190.0,
        ]
    )
