"""
Run AIFS2 wave analysis for all storms in one operational cycle.

The wrapper discovers storms from existing operational AIFS2 track
products, validates each case with analyze_aifs2_waves.py --dry-run,
runs the wave analysis for cases that pass validation, and optionally
updates the Odalys/Polo multicycle comparison.

One storm failure does not prevent the remaining storms from running.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path


OPERATIONAL_ROOT = Path("results/operational")
ANALYZER = Path("scripts/analyze_aifs2_waves.py")
MULTICYCLE = Path("scripts/plot_multicycle_wave_comparison.py")

INIT_PATTERN = re.compile(
    r"^\d{8}T\d{6}$"
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run operational AIFS2 wave analysis for every "
            "storm available in one initialization cycle."
        )
    )

    parser.add_argument(
        "--init",
        required=True,
        help=(
            "Operational initialization in "
            "YYYYMMDDTHHMMSS format."
        ),
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Discover storms and run analyzer preflight only. "
            "Do not generate wave products."
        ),
    )

    parser.add_argument(
        "--skip-comparison",
        action="store_true",
        help=(
            "Do not update the multicycle wave comparison "
            "after successful analyses."
        ),
    )

    return parser.parse_args()


def validate_init(init):
    if not INIT_PATTERN.fullmatch(init):
        raise ValueError(
            "Initialization must use YYYYMMDDTHHMMSS format: "
            f"{init!r}"
        )


def discover_storms(init):
    """
    Discover storms with an AIFS2 operational track for this cycle.

    Expected structure:

        results/operational/<storm>_<init>/aifs2/aifs2_track.csv
    """
    suffix = f"_{init}"
    storms = []

    pattern = (
        f"*_{init}/aifs2/aifs2_track.csv"
    )

    for track_path in sorted(
        OPERATIONAL_ROOT.glob(pattern)
    ):
        case_dir = track_path.parent.parent
        case_name = case_dir.name

        if not case_name.endswith(suffix):
            continue

        storm = case_name[
            :-len(suffix)
        ]

        if not storm:
            continue

        storms.append(storm)

    return sorted(set(storms))


def run_command(command):
    """
    Run one subprocess while streaming output to the terminal.
    """
    result = subprocess.run(
        command,
        check=False,
    )

    return result.returncode


def analyzer_command(storm, init, dry_run=False):
    command = [
        sys.executable,
        str(ANALYZER),
        "--storm",
        storm,
        "--init",
        init,
    ]

    if dry_run:
        command.append("--dry-run")

    return command


def print_header(title, character="="):
    print()
    print(title)
    print(character * 72)


def main():
    args = parse_args()

    try:
        validate_init(args.init)
    except ValueError as exc:
        print(
            f"ERROR: {exc}",
            file=sys.stderr,
        )
        return 2

    storms = discover_storms(
        args.init
    )

    print_header(
        "OPERATIONAL AIFS2 WAVE ANALYSIS"
    )

    print(
        f"Initialization: {args.init}"
    )

    if not storms:
        print()
        print(
            "No operational AIFS2 storm tracks were "
            "found for this initialization."
        )
        return 1

    print()
    print("Discovered storms:")

    for storm in storms:
        print(
            f"  {storm.title()}"
        )

    results = {}

    for storm in storms:
        print_header(
            storm.upper(),
            "-",
        )

        # ----------------------------------------------------
        # Preflight
        # ----------------------------------------------------
        preflight = run_command(
            analyzer_command(
                storm,
                args.init,
                dry_run=True,
            )
        )

        if preflight != 0:
            print()
            print("Preflight : FAIL")
            print("Analysis  : SKIPPED")

            results[storm] = {
                "preflight": False,
                "analysis": False,
                "skipped": True,
            }

            continue

        print()
        print("Preflight : PASS")

        if args.dry_run:
            print("Analysis  : SKIPPED (--dry-run)")

            results[storm] = {
                "preflight": True,
                "analysis": None,
                "skipped": True,
            }

            continue

        # ----------------------------------------------------
        # Full wave analysis
        # ----------------------------------------------------
        analysis = run_command(
            analyzer_command(
                storm,
                args.init,
                dry_run=False,
            )
        )

        success = analysis == 0

        print()
        print(
            "Analysis  : "
            + (
                "PASS"
                if success
                else "FAIL"
            )
        )

        results[storm] = {
            "preflight": True,
            "analysis": success,
            "skipped": False,
        }

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------
    print_header(
        "SUMMARY"
    )

    failures = 0

    for storm in storms:
        result = results[storm]

        if not result["preflight"]:
            status = "FAILED PREFLIGHT"
            failures += 1

        elif args.dry_run:
            status = "READY"

        elif result["analysis"]:
            status = "SUCCESS"

        else:
            status = "FAILED ANALYSIS"
            failures += 1

        print(
            f"{storm.title():12s} {status}"
        )

    # --------------------------------------------------------
    # Multicycle comparison
    # --------------------------------------------------------
    if args.dry_run:
        print()
        print(
            "Multicycle comparison: "
            "SKIPPED (--dry-run)"
        )

    elif args.skip_comparison:
        print()
        print(
            "Multicycle comparison: "
            "SKIPPED (--skip-comparison)"
        )

    elif failures:
        print()
        print(
            "Multicycle comparison: "
            "SKIPPED (one or more analyses failed)"
        )

    else:
        print_header(
            "MULTICYCLE WAVE COMPARISON"
        )

        comparison = run_command(
            [
                sys.executable,
                str(MULTICYCLE),
            ]
        )

        print()

        if comparison == 0:
            print(
                "Multicycle comparison: UPDATED"
            )
        else:
            print(
                "Multicycle comparison: FAILED"
            )
            failures += 1

    if failures:
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
