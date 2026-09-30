"""Regression tests for multicycle operational wave comparison."""

from pathlib import Path


SCRIPT = Path(
    "scripts/plot_multicycle_wave_comparison.py"
)


def source():
    return SCRIPT.read_text()


def test_default_storms_preserve_odalys_polo_workflow():
    text = source()

    assert (
        'DEFAULT_STORMS = ("Odalys", "Polo")'
        in text
    )

    assert (
        "default=list(DEFAULT_STORMS)"
        in text
    )


def test_multicycle_wave_comparison_accepts_storm_cli():
    text = source()

    assert '"--storms"' in text
    assert 'nargs="+"' in text


def test_plot_columns_follow_selected_storms():
    text = source()

    assert "n_storms = len(STORMS)" in text
    assert "for column, storm in enumerate(STORMS):" in text

    # Important for the one-storm Rachel case:
    # axes must remain two-dimensional.
    assert "squeeze=False" in text


def test_output_filename_uses_selected_storms():
    text = source()

    assert 'storm_slug = "_".join(' in text
    assert 'f"{storm_slug}_wave_cycles_"' in text


def test_fixed_odalys_polo_plot_loop_removed():
    text = source()

    assert (
        'enumerate(("Odalys", "Polo"))'
        not in text
    )


def test_fixed_polo_odalys_output_name_removed():
    text = source()

    assert (
        'f"polo_odalys_wave_cycles_"'
        not in text
    )
