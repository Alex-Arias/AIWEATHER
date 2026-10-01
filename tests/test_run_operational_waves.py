"""Regression tests for operational wave dispatch."""

from pathlib import Path


SCRIPT = Path("scripts/run_operational_waves.py")


def source():
    return SCRIPT.read_text()


def test_multicycle_dispatch_forwards_discovered_storms():
    text = source()

    expected = '''        comparison = run_command(
            [
                sys.executable,
                str(MULTICYCLE),
                "--storms",
                *storms,
            ]
        )
'''

    assert expected in text


def test_driver_no_longer_describes_comparison_as_odalys_polo():
    text = source()

    assert (
        "updates the Odalys/Polo multicycle comparison."
        not in text
    )

    assert (
        "updates the multicycle comparison for the storms "
        "discovered in the current operational cycle."
        in text
    )
