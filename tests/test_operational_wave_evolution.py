"""Tests for the operational AIFS2 wave-evolution plotter."""

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import pandas as pd


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "plot_operational_wave_evolution.py"
)

SPEC = spec_from_file_location(
    "plot_operational_wave_evolution",
    SCRIPT,
)

MODULE = module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_load_diagnostics_reconstructs_valid_time(tmp_path):
    path = tmp_path / "diagnostics.csv"

    pd.DataFrame(
        {
            "lead_time_hours": [0, 6, 24],
            "max_swh_300km_m": [5.0, 5.5, 6.0],
            "mean_swh_300km_m": [3.0, 3.2, 3.5],
            "max_wind_300km_ms": [20.0, 21.0, 22.0],
            "mean_mwp_300km_s": [8.0, 8.2, 8.5],
        }
    ).to_csv(path, index=False)

    result = MODULE.load_diagnostics(
        path,
        "20260928T120000",
    )

    assert result["lead_time_hours"].tolist() == [
        0,
        6,
        24,
    ]

    expected = pd.to_datetime(
        [
            "2026-09-28 12:00:00+00:00",
            "2026-09-28 18:00:00+00:00",
            "2026-09-29 12:00:00+00:00",
        ]
    )

    assert result["valid_time"].tolist() == list(expected)


def test_load_diagnostics_sorts_lead_time(tmp_path):
    path = tmp_path / "diagnostics.csv"

    pd.DataFrame(
        {
            "lead_time_hours": [24, 0, 6],
            "max_swh_300km_m": [6.0, 5.0, 5.5],
            "mean_swh_300km_m": [3.5, 3.0, 3.2],
            "max_wind_300km_ms": [22.0, 20.0, 21.0],
            "mean_mwp_300km_s": [8.5, 8.0, 8.2],
        }
    ).to_csv(path, index=False)

    result = MODULE.load_diagnostics(
        path,
        "20260928T120000",
    )

    assert result["lead_time_hours"].tolist() == [
        0,
        6,
        24,
    ]


def test_load_diagnostics_rejects_missing_columns(tmp_path):
    path = tmp_path / "diagnostics.csv"

    pd.DataFrame(
        {
            "lead_time_hours": [0],
            "max_swh_300km_m": [5.0],
        }
    ).to_csv(path, index=False)

    try:
        MODULE.load_diagnostics(
            path,
            "20260928T120000",
        )
    except ValueError as exc:
        message = str(exc)

        assert "missing required columns" in message.lower()
        assert "max_wind_300km_ms" in message
        assert "mean_mwp_300km_s" in message
        assert "mean_swh_300km_m" in message
    else:
        raise AssertionError(
            "Expected ValueError for missing columns"
        )


def test_make_figure_creates_png_and_pdf(tmp_path):
    diagnostics = tmp_path / "diagnostics.csv"

    pd.DataFrame(
        {
            "lead_time_hours": [0, 24, 48],
            "max_swh_300km_m": [5.0, 6.0, 5.5],
            "mean_swh_300km_m": [3.0, 4.0, 3.5],
            "max_wind_300km_ms": [20.0, 22.0, 19.0],
            "mean_mwp_300km_s": [8.0, 9.0, 8.5],
        }
    ).to_csv(diagnostics, index=False)

    df = MODULE.load_diagnostics(
        diagnostics,
        "20260928T120000",
    )

    output = tmp_path / "wave_evolution.png"

    MODULE.make_figure(
        df,
        storm="Rachel",
        init="20260928T120000",
        output=output,
    )

    assert output.is_file()
    assert output.stat().st_size > 0

    pdf = output.with_suffix(".pdf")

    assert pdf.is_file()
    assert pdf.stat().st_size > 0
