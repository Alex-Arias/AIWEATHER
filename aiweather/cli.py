"""
Command-line interface for AIWeather.

The CLI is intentionally thin: it parses command-line arguments and
delegates scientific work to the tested AIWeather Python APIs.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

from aiweather import __version__
from aiweather.verification import (
    run_tc_verification_pipeline,
)


def _build_parser() -> argparse.ArgumentParser:
    """
    Build the top-level AIWeather argument parser.
    """
    parser = argparse.ArgumentParser(
        prog="aiweather",
        description=(
            "AIWeather forecasting, tropical cyclone "
            "tracking, and verification toolkit."
        ),
    )

    parser.add_argument(
        "--version",
        action="version",
        version=(
            f"%(prog)s {__version__}"
        ),
    )

    subparsers = parser.add_subparsers(
        dest="command",
        required=True,
    )

    verify_parser = subparsers.add_parser(
        "verify-tc",
        help=(
            "Run tropical cyclone tracking and "
            "IBTrACS verification."
        ),
    )

    verify_parser.add_argument(
        "--forecast",
        required=True,
        type=Path,
        help=(
            "Path to the AIWeather forecast store."
        ),
    )

    verify_parser.add_argument(
        "--sid",
        required=True,
        help=(
            "IBTrACS storm identifier."
        ),
    )

    verify_parser.add_argument(
        "--lat-min",
        required=True,
        type=float,
        help="Minimum tracking latitude.",
    )

    verify_parser.add_argument(
        "--lat-max",
        required=True,
        type=float,
        help="Maximum tracking latitude.",
    )

    verify_parser.add_argument(
        "--lon-min",
        required=True,
        type=float,
        help="Minimum tracking longitude.",
    )

    verify_parser.add_argument(
        "--lon-max",
        required=True,
        type=float,
        help="Maximum tracking longitude.",
    )

    verify_parser.add_argument(
        "--device",
        default="cpu",
        help=(
            "Tracking device, for example "
            "'cpu' or 'cuda'."
        ),
    )

    verify_parser.add_argument(
        "--minimum-overlap",
        type=int,
        default=1,
        help=(
            "Minimum overlap required for "
            "tracker-path matching."
        ),
    )

    verify_parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Directory for verification CSV products."
        ),
    )

    verify_parser.add_argument(
        "--plots",
        action="store_true",
        help=(
            "Generate standard verification plots."
        ),
    )

    verify_parser.add_argument(
        "--plot-output",
        type=Path,
        default=None,
        help=(
            "Optional directory for generated plots."
        ),
    )

    verify_parser.add_argument(
        "--field-leads",
        nargs="+",
        type=float,
        default=None,
        metavar="HOUR",
        help=(
            "Forecast lead times for the "
            "multi-panel TC field sequence."
        ),
    )

    verify_parser.add_argument(
        "--ibtracs-path",
        type=Path,
        default=None,
        help=(
            "Explicit IBTrACS CSV path. "
            "If omitted, AIWeather manages the "
            "IBTrACS cache automatically."
        ),
    )

    verify_parser.add_argument(
        "--ibtracs-basin",
        default="EP",
        help=(
            "IBTrACS basin used for automatic "
            "dataset resolution."
        ),
    )

    verify_parser.add_argument(
        "--ibtracs-cache-dir",
        type=Path,
        default=Path(
            "data/verification/ibtracs"
        ),
        help=(
            "Directory used for the local "
            "IBTrACS cache."
        ),
    )

    verify_parser.add_argument(
        "--ibtracs-max-age-hours",
        type=float,
        default=48.0,
        help=(
            "Maximum cache age before the "
            "IBTrACS dataset is refreshed."
        ),
    )

    verify_parser.add_argument(
        "--ibtracs-force-update",
        action="store_true",
        help=(
            "Force an IBTrACS refresh even when "
            "the cached file is fresh."
        ),
    )

    return parser

def _validate_verify_tc_args(
    args: argparse.Namespace,
) -> None:
    """
    Validate ``verify-tc`` command-line arguments.
    """
    if args.lat_min >= args.lat_max:
        raise ValueError(
            "--lat-min must be smaller than "
            "--lat-max."
        )

    if args.lon_min >= args.lon_max:
        raise ValueError(
            "--lon-min must be smaller than "
            "--lon-max."
        )

    if args.minimum_overlap < 1:
        raise ValueError(
            "--minimum-overlap must be at least 1."
        )

    if args.ibtracs_max_age_hours <= 0.0:
        raise ValueError(
            "--ibtracs-max-age-hours must be "
            "positive."
        )

    if args.field_leads is not None:
        if any(
            lead < 0
            for lead in args.field_leads
        ):
            raise ValueError(
                "--field-leads cannot contain "
                "negative values."
            )

        if not args.plots:
            raise ValueError(
                "--field-leads requires --plots."
            )

def _run_verify_tc(
    args: argparse.Namespace,
) -> int:
    """
    Execute the ``verify-tc`` subcommand.
    """
    _validate_verify_tc_args(
        args
    )

    result = run_tc_verification_pipeline(
        args.forecast,
        sid=args.sid,
        lat_min=args.lat_min,
        lat_max=args.lat_max,
        lon_min=args.lon_min,
        lon_max=args.lon_max,
        ibtracs_path=args.ibtracs_path,
        device=args.device,
        minimum_overlap=(
            args.minimum_overlap
        ),
        output_dir=args.output,
        generate_plots=args.plots,
        plot_output_dir=(
            args.plot_output
        ),
        field_lead_times=(
            args.field_leads
        ),
        ibtracs_basin=(
            args.ibtracs_basin
        ),
        ibtracs_cache_dir=(
            args.ibtracs_cache_dir
        ),
        ibtracs_max_age_hours=(
            args.ibtracs_max_age_hours
        ),
        ibtracs_force_update=(
            args.ibtracs_force_update
        ),
    )

    print()
    print(
        "Tropical cyclone verification complete."
    )

    print(
        f"Storm SID: {args.sid}"
    )

    if result.output_dir is not None:
        print(
            "Verification output:",
            result.output_dir,
        )

    if result.plot_paths:
        print()
        print(
            "Generated plots:"
        )

        for (
            name,
            path,
        ) in result.plot_paths.items():
            print(
                f"  {name:24s} {path}"
            )

    print()
    print(
        "Verification summary:"
    )

    for (
        name,
        verification,
    ) in (
        result.verification
        .verifications
        .items()
    ):
        print(
            f"  {name:8s} "
            f"n={verification.overlap_count:3d} "
            f"mean={verification.mean_track_error_km:8.2f} km "
            f"rmse={verification.rmse_track_error_km:8.2f} km"
        )

    return 0


def main(
    argv: list[str] | None = None,
) -> int:
    """
    Run the AIWeather command-line interface.
    """
    parser = _build_parser()

    args = parser.parse_args(
        argv
    )

    if args.command == "verify-tc":
        try:
            return _run_verify_tc(
                args
            )
        except ValueError as exc:
            parser.error(
                str(exc)
            )

    parser.error(
        f"unsupported command: {args.command}"
    )

    return 2


if __name__ == "__main__":
    sys.exit(
        main()
    )
