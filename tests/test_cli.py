from pathlib import Path
from types import SimpleNamespace

import pytest

from aiweather.cli import (
    main,
)


def make_pipeline_result():
    verification = SimpleNamespace(
        verifications={
            "native": SimpleNamespace(
                overlap_count=31,
                mean_track_error_km=544.17,
                rmse_track_error_km=590.70,
            ),
            "wuduan": SimpleNamespace(
                overlap_count=36,
                mean_track_error_km=491.97,
                rmse_track_error_km=548.24,
            ),
            "vitart": SimpleNamespace(
                overlap_count=5,
                mean_track_error_km=411.09,
                rmse_track_error_km=415.44,
            ),
        }
    )

    return SimpleNamespace(
        verification=verification,
        output_dir=Path(
            "results/verification/case"
        ),
        plot_paths={
            "track_map": Path(
                "results/verification/case/"
                "plots/track_map.png"
            ),
        },
    )


def test_cli_help(capsys):
    with pytest.raises(
        SystemExit,
    ) as exc:
        main(
            [
                "--help",
            ]
        )

    assert exc.value.code == 0

    output = capsys.readouterr().out

    assert "verify-tc" in output
    assert "AIWeather" in output


def test_cli_verify_tc_help(capsys):
    with pytest.raises(
        SystemExit,
    ) as exc:
        main(
            [
                "verify-tc",
                "--help",
            ]
        )

    assert exc.value.code == 0

    output = capsys.readouterr().out

    assert "--forecast" in output
    assert "--sid" in output
    assert "--plots" in output
    assert "--field-leads" in output
    assert "--ibtracs-path" in output


def test_cli_verify_tc_calls_pipeline(
    monkeypatch,
    capsys,
):
    captured = {}

    def fake_pipeline(
        forecast_path,
        **kwargs,
    ):
        captured[
            "forecast_path"
        ] = forecast_path

        captured[
            "kwargs"
        ] = kwargs

        return make_pipeline_result()

    monkeypatch.setattr(
        "aiweather.cli."
        "run_tc_verification_pipeline",
        fake_pipeline,
    )

    status = main(
        [
            "verify-tc",
            "--forecast",
            "forecast.zarr",
            "--sid",
            "2026204N08267",
            "--lat-min",
            "5",
            "--lat-max",
            "35",
            "--lon-min",
            "-130",
            "--lon-max",
            "-90",
            "--device",
            "cuda",
            "--minimum-overlap",
            "3",
            "--output",
            "results/verification/case",
            "--plots",
            "--field-leads",
            "54",
            "78",
            "96",
            "120",
        ]
    )

    assert status == 0

    assert captured[
        "forecast_path"
    ] == Path(
        "forecast.zarr"
    )

    kwargs = captured[
        "kwargs"
    ]

    assert kwargs[
        "sid"
    ] == "2026204N08267"

    assert kwargs[
        "lat_min"
    ] == pytest.approx(
        5.0
    )

    assert kwargs[
        "lat_max"
    ] == pytest.approx(
        35.0
    )

    assert kwargs[
        "lon_min"
    ] == pytest.approx(
        -130.0
    )

    assert kwargs[
        "lon_max"
    ] == pytest.approx(
        -90.0
    )

    assert kwargs[
        "device"
    ] == "cuda"

    assert kwargs[
        "minimum_overlap"
    ] == 3

    assert kwargs[
        "generate_plots"
    ] is True

    assert kwargs[
        "field_lead_times"
    ] == [
        54.0,
        78.0,
        96.0,
        120.0,
    ]

    output = capsys.readouterr().out

    assert (
        "Tropical cyclone verification complete."
        in output
    )

    assert (
        "2026204N08267"
        in output
    )

    assert (
        "track_map"
        in output
    )

    assert (
        "native"
        in output
    )


def test_cli_verify_tc_defaults(
    monkeypatch,
):
    captured = {}

    def fake_pipeline(
        forecast_path,
        **kwargs,
    ):
        captured[
            "kwargs"
        ] = kwargs

        return SimpleNamespace(
            verification=SimpleNamespace(
                verifications={}
            ),
            output_dir=None,
            plot_paths={},
        )

    monkeypatch.setattr(
        "aiweather.cli."
        "run_tc_verification_pipeline",
        fake_pipeline,
    )

    status = main(
        [
            "verify-tc",
            "--forecast",
            "forecast.zarr",
            "--sid",
            "2026204N08267",
            "--lat-min",
            "5",
            "--lat-max",
            "35",
            "--lon-min",
            "-130",
            "--lon-max",
            "-90",
        ]
    )

    assert status == 0

    kwargs = captured[
        "kwargs"
    ]

    assert kwargs[
        "device"
    ] == "cpu"

    assert kwargs[
        "minimum_overlap"
    ] == 1

    assert kwargs[
        "generate_plots"
    ] is False

    assert kwargs[
        "field_lead_times"
    ] is None

    assert kwargs[
        "ibtracs_path"
    ] is None

    assert kwargs[
        "ibtracs_basin"
    ] == "EP"

    assert kwargs[
        "ibtracs_max_age_hours"
    ] == pytest.approx(
        48.0
    )

    assert kwargs[
        "ibtracs_force_update"
    ] is False


def test_cli_verify_tc_explicit_ibtracs(
    monkeypatch,
):
    captured = {}

    def fake_pipeline(
        forecast_path,
        **kwargs,
    ):
        captured[
            "kwargs"
        ] = kwargs

        return SimpleNamespace(
            verification=SimpleNamespace(
                verifications={}
            ),
            output_dir=None,
            plot_paths={},
        )

    monkeypatch.setattr(
        "aiweather.cli."
        "run_tc_verification_pipeline",
        fake_pipeline,
    )

    status = main(
        [
            "verify-tc",
            "--forecast",
            "forecast.zarr",
            "--sid",
            "2026204N08267",
            "--lat-min",
            "5",
            "--lat-max",
            "35",
            "--lon-min",
            "-130",
            "--lon-max",
            "-90",
            "--ibtracs-path",
            "frozen_ibtracs.csv",
            "--ibtracs-basin",
            "EP",
            "--ibtracs-force-update",
        ]
    )

    assert status == 0

    kwargs = captured[
        "kwargs"
    ]

    assert kwargs[
        "ibtracs_path"
    ] == Path(
        "frozen_ibtracs.csv"
    )

    assert kwargs[
        "ibtracs_force_update"
    ] is True

def test_cli_version(capsys):
    from aiweather import (
        __version__,
    )

    with pytest.raises(
        SystemExit,
    ) as exc:
        main(
            [
                "--version",
            ]
        )

    assert exc.value.code == 0

    output = (
        capsys
        .readouterr()
        .out
        .strip()
    )

    assert output == (
        f"aiweather {__version__}"
    )