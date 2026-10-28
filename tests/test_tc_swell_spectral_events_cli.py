"""Regression tests for experimental Phase B2 spectral-event CLI."""

from argparse import Namespace
from pathlib import Path
import sys

import pandas as pd
import pytest

from scripts import build_tc_swell_advisory_b2_candidate as advisory


INIT = "20261005T120000"


def make_args(**overrides):
    values = {
        "mode": "spectral-event",
        "storm": "rachel",
        "init": INIT,
        "tracker": "wuduan",
        "output": None,
    }
    values.update(overrides)
    return Namespace(**values)


def make_inputs(tmp_path):
    root = (
        tmp_path
        / "results"
        / "operational"
        / f"rachel_{INIT}"
        / "advisory"
    )
    root.mkdir(parents=True)

    events = pd.DataFrame({
        "sector": ["BC North"],
        "band": ["long"],
        "onset_lead_h": [6],
        "end_lead_h": [18],
        "duration_h": [12],
        "primary_peak_lead_h": [12],
        "primary_peak_band_hs_m": [1.5],
        "n_constituent_peaks": [1],
        "constituent_peak_leads_h": ["12"],
        "left_censored": [False],
        "right_censored": [False],
        "ambiguous_overlap": [False],
        "minimum_interpeak_valley_ratio": [0.5],
    })

    quality = events.copy()
    quality["record_type"] = "segmented_episode"
    quality["temporal_resolution"] = "resolved"
    quality["boundary_change_24h_m"] = float("nan")
    quality["boundary_monotonic_24h"] = False
    quality["boundary_continuation_candidate"] = False

    times = pd.to_datetime([
        "2026-10-05 12:00:00",
        "2026-10-05 18:00:00",
        "2026-10-06 00:00:00",
        "2026-10-06 06:00:00",
    ])

    b1 = pd.DataFrame({
        "sector": ["BC North"] * 4,
        "valid_time": times,
        "lead_time_hours": [0, 6, 12, 18],
        "tc_guidance_available": [True] * 4,
        "tc_support_status": ["supported"] * 4,
        "tc_jointly_supported": [True] * 4,
        "tc_latitude": [20.0] * 4,
        "tc_longitude": [-120.0] * 4,
        "tc_pressure_hpa": [990.0] * 4,
        "tc_wind_ms": [25.0] * 4,
        "tc_r67_km": [80.0] * 4,
        "tc_r90_km": [120.0] * 4,
        "spectral_sampling_supported": [True] * 4,
        "directional_qc_supported": [True] * 4,
    })

    events_path = root / "rachel_aifs2_spectral_events.csv"
    quality_path = root / "rachel_aifs2_spectral_quality.csv"
    b1_path = root / "rachel_wuduan_coastal_wave_guidance.csv"

    events.to_csv(events_path, index=False)
    quality.to_csv(quality_path, index=False)
    b1.to_csv(b1_path, index=False)

    return root, events_path, quality_path, b1_path


def run_cli(tmp_path, monkeypatch, **overrides):
    monkeypatch.chdir(tmp_path)
    return advisory.run_spectral_event_mode(
        make_args(**overrides)
    )


def test_successful_spectral_event_integration(
    tmp_path, monkeypatch
):
    root, *_ = make_inputs(tmp_path)

    run_cli(tmp_path, monkeypatch)

    output = root / (
        "rachel_wuduan_spectral_event_guidance.csv"
    )

    assert output.is_file()

    result = pd.read_csv(output)

    assert len(result) == 1
    assert "event_tc_horizon_class" in result.columns
    assert "peak_tc_support_status" in result.columns


def test_incorrect_initialization_rejected(
    tmp_path, monkeypatch
):
    import shutil

    original_root, *_ = make_inputs(tmp_path)

    incorrect_init = "20261005T180000"

    incorrect_root = (
        tmp_path
        / "results"
        / "operational"
        / f"rachel_{incorrect_init}"
        / "advisory"
    )

    shutil.copytree(original_root, incorrect_root)

    with pytest.raises(ValueError, match="timestamps"):
        run_cli(
            tmp_path,
            monkeypatch,
            init=incorrect_init,
        )

    assert not (
        incorrect_root
        / "rachel_wuduan_spectral_event_guidance.csv"
    ).exists()


@pytest.mark.parametrize(
    "input_index",
    [1, 2, 3],
)
def test_missing_input_rejected(
    tmp_path, monkeypatch, input_index
):
    paths = make_inputs(tmp_path)

    paths[input_index].unlink()

    with pytest.raises(FileNotFoundError):
        run_cli(tmp_path, monkeypatch)


def test_existing_output_not_overwritten(
    tmp_path, monkeypatch
):
    root, *_ = make_inputs(tmp_path)

    output = root / (
        "rachel_wuduan_spectral_event_guidance.csv"
    )

    output.write_text("PROTECTED\\n")

    with pytest.raises(SystemExit, match="already exists"):
        run_cli(tmp_path, monkeypatch)

    assert output.read_text() == "PROTECTED\\n"


def test_explicit_output_path(
    tmp_path, monkeypatch
):
    make_inputs(tmp_path)

    output = tmp_path / "custom_b2.csv"

    run_cli(
        tmp_path,
        monkeypatch,
        output=output,
    )

    assert output.is_file()
    assert len(pd.read_csv(output)) == 1


def test_invalid_initialization_format_rejected(
    tmp_path, monkeypatch
):
    make_inputs(tmp_path)

    with pytest.raises((ValueError, TypeError)):
        run_cli(
            tmp_path,
            monkeypatch,
            init="invalid",
        )


def test_spectral_event_mode_parsing(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "build_tc_swell_advisory_b2_candidate.py",
            "--mode",
            "spectral-event",
            "--storm",
            "rachel",
            "--init",
            INIT,
        ],
    )

    args = advisory.parse_args()

    assert args.mode == "spectral-event"
    assert args.storm == "rachel"
    assert args.init == INIT


def test_default_mode_remains_tc(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "build_tc_swell_advisory_b2_candidate.py",
            "--storm",
            "rachel",
            "--init",
            INIT,
        ],
    )

    args = advisory.parse_args()

    assert args.mode == "tc"


def test_no_physical_attribution_claim(
    tmp_path, monkeypatch, capsys
):
    make_inputs(tmp_path)

    run_cli(tmp_path, monkeypatch)

    output = capsys.readouterr().out.lower()

    assert "temporal coincidence" in output
    assert "does not establish" in output
    assert "attribution" in output
