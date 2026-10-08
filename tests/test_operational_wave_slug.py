"""Regression tests for operational wave storm-name slugging."""

from pathlib import Path


SCRIPT = Path("scripts/analyze_aifs2_waves.py")


def source():
    return SCRIPT.read_text()


def test_operational_wave_analyzer_defines_storm_slug():
    text = source()

    assert '''def storm_slug(name):
    """Return the canonical filesystem slug for a storm name."""
    return (
        name.lower()
        .replace(" ", "_")
    )
''' in text


def test_operational_case_uses_canonical_storm_slug():
    text = source()

    assert "storm_key = storm_slug(storm)" in text

    assert (
        '"storm_name": storm_key.replace("_", " ").title(),'
        in text
    )

    assert '"storm_slug": storm_key,' in text


def test_operational_wave_output_key_uses_slug():
    text = source()

    assert '''STORM_KEY = (
        f"{CASE.get('experiment_slug', storm_slug(args.storm))}_"
        f"{args.init[:8]}"
    )
''' in text


def test_operational_wave_supports_experiment_slug():
    text = source()

    assert '"--experiment-slug"' in text
    assert "experiment_slug=args.experiment_slug" in text
    assert '"experiment_slug": experiment_key,' in text
    assert 'f"{experiment_key}_{init}"' in text


def test_operational_wave_preserves_storm_identity():
    text = source()

    assert '"storm_name": storm_key.replace("_", " ").title(),' in text
    assert '"storm_slug": storm_key,' in text
    assert '"experiment_slug": experiment_key,' in text


def test_operational_wave_preserves_default_slug():
    text = source()

    assert '''experiment_key = (
        storm_key
        if experiment_slug is None
        else experiment_slug
    )''' in text


def test_operational_product_filenames_use_slug():
    text = source()

    assert (
        "CASE.get('storm_slug', STORM_NAME.lower())"
        in text
    )

    assert (
        'f"aifs2_{STORM_NAME.lower()}_"'
        not in text
    )
