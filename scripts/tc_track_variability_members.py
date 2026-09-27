"""
Build aligned AIWeather tropical-cyclone track members.

The resulting CSV is the input expected by tc_track_variability.py.

Forecast tracks are aligned by VALID TIME rather than forecast lead.
Only the common 6-hour synoptic grid is retained.

Example
-------
python scripts/tc_track_variability_members.py \
    --storm Odalys \
    --tracker wuduan \
    --operational-root results/operational \
    --output results/operational/odalys_multicycle
"""

import argparse
from pathlib import Path

import pandas as pd


DEFAULT_MODELS = [
    "graphcast",
    "aifs2",
    "pangu3",
    "pangu6",
]


def parse_args():

    parser = argparse.ArgumentParser(
        description=(
            "Build aligned multimodel/multicycle tropical-cyclone "
            "track members for AIWeather variability diagnostics."
        )
    )

    parser.add_argument(
        "--storm",
        required=True,
        help="Storm name, for example Polo or Odalys.",
    )

    parser.add_argument(
        "--tracker",
        default="wuduan",
        help="Tracker name. Default: wuduan.",
    )

    parser.add_argument(
        "--operational-root",
        type=Path,
        default=Path("results/operational"),
        help="Root containing storm_cycle operational directories.",
    )

    parser.add_argument(
        "--models",
        nargs="+",
        default=DEFAULT_MODELS,
        help="Models to include.",
    )

    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Output directory.",
    )

    return parser.parse_args()


def discover_cycles(root, storm):

    prefix = f"{storm.lower()}_"

    directories = sorted(
        p
        for p in root.iterdir()
        if p.is_dir()
        and p.name.lower().startswith(prefix)
    )

    cycles = []

    for directory in directories:

        cycle_string = directory.name[len(prefix):]

        try:
            cycle_time = pd.to_datetime(
                cycle_string,
                format="%Y%m%dT%H%M%S",
            )
        except ValueError:
            continue

        cycles.append(
            (cycle_time, directory)
        )

    return cycles


def load_track(path, cycle_number, cycle_time, model):

    df = pd.read_csv(
        path,
        parse_dates=["valid_time"],
    )

    required = {
        "lead_time_hours",
        "valid_time",
        "latitude",
        "longitude",
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"{path} missing required columns: "
            f"{sorted(missing)}"
        )

    out = df[
        [
            "lead_time_hours",
            "valid_time",
            "latitude",
            "longitude",
        ]
    ].copy()

    out["cycle"] = cycle_number
    out["cycle_time"] = cycle_time
    out["model"] = model

    # Normalize valid times to the 6-hour synoptic grid.
    #
    # We retain only actual 6-hour forecast records; we do not
    # interpolate tracks onto missing times.
    valid = (
        out["valid_time"].dt.minute.eq(0)
        & out["valid_time"].dt.second.eq(0)
        & out["valid_time"].dt.hour.mod(6).eq(0)
    )

    out = out.loc[valid].copy()

    out = out.dropna(
        subset=[
            "latitude",
            "longitude",
        ]
    )

    return out


def main():

    args = parse_args()

    args.output.mkdir(
        parents=True,
        exist_ok=True,
    )

    cycles = discover_cycles(
        args.operational_root,
        args.storm,
    )

    if not cycles:
        raise SystemExit(
            f"No operational cycles found for {args.storm}"
        )

    members = []
    availability = []

    print("=" * 78)
    print(
        f" AIWEATHER TRACK MEMBER BUILDER — "
        f"{args.storm.upper()}"
    )
    print("=" * 78)

    print()
    print("Operational cycles:")

    for cycle_number, (cycle_time, directory) in enumerate(
        cycles,
        start=1,
    ):

        print(
            f"  {cycle_number}: "
            f"{cycle_time:%Y-%m-%d %H UTC}"
        )

        for model in args.models:

            track_file = (
                directory
                / model
                / f"{model}_{args.tracker}_track.csv"
            )

            if not track_file.exists():

                print(
                    f"    MISSING {model}: "
                    f"{track_file}"
                )

                continue

            track = load_track(
                track_file,
                cycle_number,
                cycle_time,
                model,
            )

            print(
                f"    {model:10s} "
                f"{len(track):3d} records"
            )

            members.append(track)

    if not members:
        raise SystemExit(
            "No compatible tracker files found."
        )

    members = pd.concat(
        members,
        ignore_index=True,
    )

    members = members.sort_values(
        [
            "valid_time",
            "cycle",
            "model",
        ]
    ).reset_index(drop=True)

    # Protect against accidental duplicate model/cycle/time records.
    duplicated = members.duplicated(
        subset=[
            "valid_time",
            "cycle",
            "model",
        ],
        keep=False,
    )

    if duplicated.any():
        duplicate_rows = members.loc[
            duplicated,
            [
                "valid_time",
                "cycle",
                "model",
            ],
        ]

        raise ValueError(
            "Duplicate cycle/model/valid-time records found:\n"
            + duplicate_rows.to_string(index=False)
        )

    # Availability by valid time.
    availability = (
        members.groupby(
            "valid_time",
            as_index=False,
        )
        .agg(
            n_members=("model", "size"),
            n_cycles=("cycle", "nunique"),
            n_models=("model", "nunique"),
            min_lead_h=("lead_time_hours", "min"),
            max_lead_h=("lead_time_hours", "max"),
        )
        .sort_values("valid_time")
        .reset_index(drop=True)
    )

    stem = (
        f"{args.storm.lower()}_"
        f"{args.tracker.lower()}"
    )

    members_file = (
        args.output
        / f"{stem}_lagged_members.csv"
    )

    availability_file = (
        args.output
        / f"{stem}_availability.csv"
    )

    members.to_csv(
        members_file,
        index=False,
    )

    availability.to_csv(
        availability_file,
        index=False,
    )

    print()
    print("-" * 78)
    print("SUMMARY")
    print("-" * 78)

    print(
        "members rows :",
        len(members),
    )

    print(
        "valid times  :",
        members["valid_time"].nunique(),
    )

    print(
        "cycles       :",
        members["cycle"].nunique(),
    )

    print(
        "models       :",
        members["model"].nunique(),
    )

    print()
    print("Member counts by model:")
    print(
        members.groupby("model")
        .size()
        .to_string()
    )

    print()
    print("Member counts by cycle:")
    print(
        members.groupby("cycle")
        .size()
        .to_string()
    )

    print()
    print("Availability by valid time:")
    print(
        availability.to_string(
            index=False
        )
    )

    print()
    print("Files:")
    print(members_file)
    print(availability_file)

    print()
    print("DONE")


if __name__ == "__main__":
    main()
