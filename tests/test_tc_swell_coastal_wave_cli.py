"""CLI regression tests for experimental coastal-wave guidance."""

from argparse import Namespace
from pathlib import Path
import sys

import pandas as pd
import pytest

from scripts import build_tc_swell_advisory as advisory


INIT = "20261005T120000"


def make_inputs(tmp_path, *, storm="rachel", tracker="wuduan"):
    root = (
        tmp_path
        / "results"
        / "operational"
        / f"{storm}_{INIT}"
        / "advisory"
    )
    root.mkdir(parents=True)

    times = pd.to_datetime([
        "2026-10-05 12:00:00",
        "2026-10-05 18:00:00",
        "2026-10-06 00:00:00",
    ])

    wave = pd.DataFrame({
        "sector": ["BC North"] * 3,
        "valid_time": times,
        "lead_time_hours": [0, 6, 12],
        "swh_mean_m": [1.2, 1.5, 1.8],
    })

    tc = pd.DataFrame({
        "storm": [storm, storm],
        "tracker": [tracker, tracker],
        "valid_time": times[:2],
        "lead_time_hours": [0, 6],
        "tc_jointly_supported": [True, False],
        "tc_support_status": ["supported", "limited"],
        "consensus_latitude": [20.0, 20.5],
        "consensus_longitude": [-120.0, -120.5],
        "pressure_mean_hpa": [990.0, 992.0],
        "wind_mean_ms": [25.0, 23.0],
        "r67_km": [80.0, 90.0],
        "r90_km": [120.0, 140.0],
    })

    wave_path = root / f"{storm}_aifs2_spectral_guidance.csv"
    tc_path = root / f"{storm}_{tracker}_integrated_tc_guidance.csv"

    wave.to_csv(wave_path, index=False)
    tc.to_csv(tc_path, index=False)

    return root, wave_path, tc_path


def make_args(**overrides):
    values = {
        "mode": "coastal-wave",
        "storm": "rachel",
        "init": INIT,
        "tracker": "wuduan",
        "output": None,
    }
    values.update(overrides)
    return Namespace(**values)


def run_cli(tmp_path, monkeypatch, **overrides):
    monkeypatch.chdir(tmp_path)
    return advisory.run_coastal_wave_mode(
        make_args(**overrides)
    )


def test_successful_cli_integration(tmp_path, monkeypatch):
    root, _, _ = make_inputs(tmp_path)

    run_cli(tmp_path, monkeypatch)

    output = root / "rachel_wuduan_coastal_wave_guidance.csv"

    assert output.is_file()

    result = pd.read_csv(output)

    assert len(result) == 3
    assert result["tc_guidance_available"].tolist() == [
        True, True, False
    ]
    assert result["swh_mean_m"].tolist() == [1.2, 1.5, 1.8]
    assert pd.isna(result.loc[2, "tc_latitude"])
    assert result.loc[2, "tc_support_status"] == "unavailable"


def test_incorrect_initialization_rejected(tmp_path, monkeypatch):
    root, wave_path, tc_path = make_inputs(tmp_path)

    wrong_init = "20261006T120000"

    wrong_root = (
        tmp_path
        / "results"
        / "operational"
        / f"rachel_{wrong_init}"
        / "advisory"
    )

    wrong_root.mkdir(parents=True)

    # Preserve the original forecast timestamps deliberately.
    # Only the requested initialization and input directory differ.
    wrong_wave = wrong_root / wave_path.name
    wrong_tc = wrong_root / tc_path.name

    wrong_wave.write_bytes(wave_path.read_bytes())
    wrong_tc.write_bytes(tc_path.read_bytes())

    output = wrong_root / "wrong_init.csv"

    with pytest.raises(ValueError, match="requested initialization"):
        run_cli(
            tmp_path,
            monkeypatch,
            init=wrong_init,
            output=output,
        )

    assert not output.exists()

    # The original synthetic inputs must remain unchanged.
    assert wave_path.is_file()
    assert tc_path.is_file()


@pytest.mark.parametrize("column", ["storm", "tracker"])
def test_missing_provenance_column_rejected(
    tmp_path, monkeypatch, column
):
    root, _, tc_path = make_inputs(tmp_path)

    tc = pd.read_csv(tc_path)
    tc = tc.drop(columns=[column])
    tc.to_csv(tc_path, index=False)

    with pytest.raises(ValueError, match="missing provenance"):
        run_cli(tmp_path, monkeypatch)

    assert not (
        root / "rachel_wuduan_coastal_wave_guidance.csv"
    ).exists()


@pytest.mark.parametrize("column", ["storm", "tracker"])
def test_missing_provenance_value_rejected(
    tmp_path, monkeypatch, column
):
    root, _, tc_path = make_inputs(tmp_path)

    tc = pd.read_csv(tc_path)
    tc.loc[0, column] = None
    tc.to_csv(tc_path, index=False)

    with pytest.raises(ValueError, match="missing"):
        run_cli(tmp_path, monkeypatch)

    assert not (
        root / "rachel_wuduan_coastal_wave_guidance.csv"
    ).exists()


@pytest.mark.parametrize(
    "column,value",
    [
        ("storm", "another_storm"),
        ("tracker", "native"),
    ],
)
def test_mismatched_provenance_rejected(
    tmp_path, monkeypatch, column, value
):
    root, _, tc_path = make_inputs(tmp_path)

    tc = pd.read_csv(tc_path)
    tc.loc[0, column] = value
    tc.to_csv(tc_path, index=False)

    with pytest.raises(ValueError, match="identifier mismatch"):
        run_cli(tmp_path, monkeypatch)

    assert not (
        root / "rachel_wuduan_coastal_wave_guidance.csv"
    ).exists()


@pytest.mark.parametrize("missing", ["wave", "tc"])
def test_missing_input_rejected(tmp_path, monkeypatch, missing):
    root, wave_path, tc_path = make_inputs(tmp_path)

    path = wave_path if missing == "wave" else tc_path
    path.unlink()

    with pytest.raises(FileNotFoundError):
        run_cli(tmp_path, monkeypatch)

    assert not (
        root / "rachel_wuduan_coastal_wave_guidance.csv"
    ).exists()


def test_existing_output_not_overwritten(tmp_path, monkeypatch):
    root, _, _ = make_inputs(tmp_path)

    output = root / "rachel_wuduan_coastal_wave_guidance.csv"
    output.write_text("PROTECTED\\n")

    with pytest.raises(SystemExit, match="already exists"):
        run_cli(tmp_path, monkeypatch)

    assert output.read_text() == "PROTECTED\\n"


def test_explicit_output_path(tmp_path, monkeypatch):
    make_inputs(tmp_path)

    output = tmp_path / "custom_b1.csv"

    run_cli(
        tmp_path,
        monkeypatch,
        output=output,
    )

    assert output.is_file()
    assert len(pd.read_csv(output)) == 3


def test_phase_a_remains_default(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "build_tc_swell_advisory",
            "--storm",
            "rachel",
            "--init",
            INIT,
        ],
    )

    args = advisory.parse_args()

    assert args.mode == "tc"
    assert args.tracker == "wuduan"


def test_coastal_wave_mode_parsing(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "build_tc_swell_advisory",
            "--mode",
            "coastal-wave",
            "--storm",
            "rachel",
            "--init",
            INIT,
        ],
    )

    args = advisory.parse_args()

    assert args.mode == "coastal-wave"
