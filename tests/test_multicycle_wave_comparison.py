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

    assert 'storm_slug = (' in text
    assert 'EXPERIMENT_SLUG' in text
    assert 'else "_".join(' in text
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



def test_multicycle_wave_supports_persistent_experiment_slug():
    text = source()

    assert '"--experiment-slug"' in text
    assert 'directory_slug = (' in text
    assert 'EXPERIMENT_SLUG' in text


def test_multicycle_wave_discovers_storm_and_experiment_filenames():
    text = source()

    assert 'diagnostic_candidates = [' in text
    assert 'f"aifs2_{directory_slug}_"' in text
    assert 'f"aifs2_{storm_lower}_"' in text
    assert 'existing_diagnostics[0]' in text


def test_multicycle_wave_uses_exact_initialization_metadata():
    text = source()

    assert 'usecols=["case_id"]' in text
    assert 'timestamp_string = case_id[len(prefix):]' in text
    assert 'format="%Y%m%dT%H%M%S"' in text
    assert 'INIT_HOUR_UTC = 12' not in text
