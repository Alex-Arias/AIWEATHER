from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd
import pytest


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "plot_operational_tc_verification.py"
)

SPEC = importlib.util.spec_from_file_location(
    "plot_operational_tc_verification",
    SCRIPT_PATH,
)

MODULE = importlib.util.module_from_spec(
    SPEC
)

assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def _write_common_table(
    directory: Path,
    model: str,
    valid_times: list[str],
    lead_times: list[int],
) -> Path:

    directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    n = len(valid_times)

    table = pd.DataFrame(
        {
            "lead_time_hours": lead_times,
            "valid_time": valid_times,
            "track_error_km": [
                10.0 + index
                for index in range(n)
            ],
            "pressure_error_pa": [
                100.0 + index
                for index in range(n)
            ],
            "wind_error_ms": [
                -5.0 + index
                for index in range(n)
            ],
        }
    )

    path = (
        directory
        / f"{model}_common_verification.csv"
    )

    table.to_csv(
        path,
        index=False,
    )

    return path


def test_load_common_verification_discovers_models(
    tmp_path,
):
    valid_times = [
        "2026-09-10T12:00:00",
        "2026-09-10T18:00:00",
    ]

    lead_times = [
        0,
        6,
    ]

    for model in [
        "pangu6",
        "aifs2",
        "graphcast",
        "pangu3",
    ]:
        _write_common_table(
            tmp_path,
            model,
            valid_times,
            lead_times,
        )

    data = MODULE.load_common_verification(
        tmp_path
    )

    assert list(data) == [
        "graphcast",
        "aifs2",
        "pangu3",
        "pangu6",
    ]

    MODULE.validate_common_times(
        data
    )


def test_validate_common_times_rejects_mismatch(
    tmp_path,
):
    _write_common_table(
        tmp_path,
        "aifs2",
        [
            "2026-09-10T12:00:00",
            "2026-09-10T18:00:00",
        ],
        [
            0,
            6,
        ],
    )

    _write_common_table(
        tmp_path,
        "graphcast",
        [
            "2026-09-10T12:00:00",
            "2026-09-11T00:00:00",
        ],
        [
            0,
            12,
        ],
    )

    data = MODULE.load_common_verification(
        tmp_path
    )

    with pytest.raises(
        ValueError,
        match=(
            "do not contain identical exact "
            "common valid times"
        ),
    ):
        MODULE.validate_common_times(
            data
        )


def test_plot_functions_create_png_and_pdf(
    tmp_path,
):
    input_dir = (
        tmp_path
        / "verification"
    )

    output_dir = (
        tmp_path
        / "plots"
    )

    valid_times = [
        "2026-09-10T12:00:00",
        "2026-09-10T18:00:00",
        "2026-09-11T00:00:00",
    ]

    lead_times = [
        0,
        6,
        12,
    ]

    for model in [
        "graphcast",
        "aifs2",
        "pangu3",
        "pangu6",
    ]:
        _write_common_table(
            input_dir,
            model,
            valid_times,
            lead_times,
        )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    data = MODULE.load_common_verification(
        input_dir
    )

    MODULE.validate_common_times(
        data
    )

    MODULE.plot_track_error(
        data,
        output_dir,
        "Test Storm",
    )

    MODULE.plot_pressure_error(
        data,
        output_dir,
        "Test Storm",
    )

    MODULE.plot_wind_error(
        data,
        output_dir,
        "Test Storm",
    )

    expected = {
        "track_error_common.png",
        "track_error_common.pdf",
        "pressure_error_common.png",
        "pressure_error_common.pdf",
        "wind_error_common.png",
        "wind_error_common.pdf",
    }

    assert {
        path.name
        for path in output_dir.iterdir()
        if path.is_file()
    } == expected

    for name in expected:
        assert (
            output_dir / name
        ).stat().st_size > 0
