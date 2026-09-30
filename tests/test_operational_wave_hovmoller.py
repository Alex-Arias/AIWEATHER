"""Regression tests for the operational AIFS2 wave Hovmoller."""

from pathlib import Path


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "plot_operational_wave_hovmoller.py"
)


def test_operational_hovmoller_script_exists():
    assert SCRIPT.is_file()


def test_operational_hovmoller_has_operational_cli():
    text = SCRIPT.read_text()

    assert '"--storm"' in text
    assert '"--init"' in text


def test_operational_hovmoller_uses_native_track_columns():
    text = SCRIPT.read_text()

    assert '"lead_time_hours"' in text
    assert '"latitude"' in text
    assert '"longitude"' in text

    assert "center_latitude" not in text
    assert "center_longitude_plot" not in text


def test_operational_hovmoller_normalizes_track_longitude():
    text = SCRIPT.read_text()

    # Operational native tracks may use 0..360 longitude,
    # while the regional plotting/extraction grid uses -180..180.
    assert "% 360.0" in text
    assert "- 180.0" in text


def test_operational_hovmoller_uses_expected_radial_domain():
    text = SCRIPT.read_text()

    assert "MAX_RADIUS_KM" in text
    assert "RADIAL_BIN_KM" in text


def test_operational_hovmoller_writes_expected_products():
    text = SCRIPT.read_text()

    assert (
        'f"{STORM_KEY}_{INIT}_wave_hovmoller.png"'
        in text
    )
    assert (
        'f"{STORM_KEY}_{INIT}_wave_hovmoller.pdf"'
        in text
    )
    assert (
        'f"{STORM_KEY}_{INIT}_wave_radial_profiles.csv"'
        in text
    )

    assert "_cycle1_wave_hovmoller" not in text
    assert "_cycle1_wave_radial_profiles" not in text


def test_operational_hovmoller_is_forecast_only():
    text = SCRIPT.read_text().lower()

    # The plotter must consume existing operational products rather
    # than invoking forecast inference or TC verification.
    assert "ibtracs" not in text
    assert "best_track" not in text
