"""
Matched-window comparison of AIWeather TC tracker variability.

Compares already-computed variability products from two trackers over
their common valid-time interval. Existing tracker products are treated
as frozen inputs and are never modified.

For each tracker this script compares:
    * historical variability (all available cycles)
    * recent variability (latest N-cycle subset)
    * latest-cycle multimodel spread

The comparison is descriptive forecast disagreement, not a calibrated
tropical-cyclone probability cone.
"""

from pathlib import Path
import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Compare two AIWeather TC track-variability products over "
            "an identical valid-time window."
        )
    )
    p.add_argument("--tracker-a-dir", required=True)
    p.add_argument("--tracker-b-dir", required=True)
    p.add_argument("--tracker-a", required=True)
    p.add_argument("--tracker-b", required=True)
    p.add_argument("--storm", required=True)
    p.add_argument("--start", required=True)
    p.add_argument("--end", required=True)
    p.add_argument("--output", required=True)
    return p.parse_args()


def load_product(directory, storm, tracker, suffix):
    path = Path(directory) / f"{storm.lower()}_{tracker}_{suffix}.csv"

    if not path.is_file():
        raise FileNotFoundError(path)

    df = pd.read_csv(path, parse_dates=["valid_time"])
    return df.sort_values("valid_time").reset_index(drop=True)


def select_window(df, start, end):
    return (
        df[df["valid_time"].between(start, end)]
        .copy()
        .reset_index(drop=True)
    )


def valid_spread(df):
    if "spread_valid" in df.columns:
        return df[df["spread_valid"].astype(bool)].copy()
    return df.copy()


def summarize(df):
    d = valid_spread(df)

    if d.empty:
        return {
            "n_times": 0,
            "mean_r67_km": np.nan,
            "mean_r90_km": np.nan,
            "mean_max_km": np.nan,
        }

    return {
        "n_times": len(d),
        "mean_r67_km": d["r67_km"].mean(),
        "mean_r90_km": d["r90_km"].mean(),
        "mean_max_km": d["max_distance_km"].mean(),
    }


def build_summary(
    tracker,
    historical,
    recent,
    latest,
    start,
    end,
):
    rows = []

    for hierarchy, df in (
        ("historical", historical),
        ("recent3", recent),
        ("latest", latest),
    ):
        s = summarize(df)

        rows.append(
            {
                "tracker": tracker,
                "hierarchy": hierarchy,
                "window_start": start,
                "window_end": end,
                **s,
            }
        )

    return rows


def contraction(hist, recent):
    h = valid_spread(hist)[
        ["valid_time", "r67_km", "r90_km"]
    ].rename(
        columns={
            "r67_km": "historical_r67_km",
            "r90_km": "historical_r90_km",
        }
    )

    r = valid_spread(recent)[
        ["valid_time", "r67_km", "r90_km"]
    ].rename(
        columns={
            "r67_km": "recent_r67_km",
            "r90_km": "recent_r90_km",
        }
    )

    m = h.merge(r, on="valid_time", how="inner")

    if m.empty:
        return np.nan, np.nan, 0

    c67 = (
        1.0
        - m["recent_r67_km"]
        / m["historical_r67_km"]
    ) * 100.0

    c90 = (
        1.0
        - m["recent_r90_km"]
        / m["historical_r90_km"]
    ) * 100.0

    return c67.mean(), c90.mean(), len(m)


def plot_metric(
    ax,
    a,
    b,
    tracker_a,
    tracker_b,
    column,
    ylabel,
    title,
):
    aa = valid_spread(a)
    bb = valid_spread(b)

    ax.plot(
        aa["valid_time"],
        aa[column],
        marker="o",
        label=tracker_a,
    )

    ax.plot(
        bb["valid_time"],
        bb[column],
        marker="o",
        label=tracker_b,
    )

    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.tick_params(axis="x", rotation=35)


def main():
    args = parse_args()

    start = pd.Timestamp(args.start)
    end = pd.Timestamp(args.end)

    # Determine the operational cycle dynamically from the frozen
    # lagged-member products rather than hard-coding a cycle number.
    latest_cycles = {}

    for tracker, tracker_dir in [
        (args.tracker_a, Path(args.tracker_a_dir)),
        (args.tracker_b, Path(args.tracker_b_dir)),
    ]:
        members_path = (
            tracker_dir
            / f"{args.storm.lower()}_{tracker}_lagged_members.csv"
        )

        if not members_path.exists():
            raise FileNotFoundError(
                f"Missing lagged-members file: {members_path}"
            )

        members = pd.read_csv(members_path)

        if "cycle" not in members.columns:
            raise ValueError(
                f"Missing cycle column in {members_path}"
            )

        latest_cycles[tracker] = int(members["cycle"].max())

    if len(set(latest_cycles.values())) != 1:
        raise ValueError(
            "Tracker latest-cycle mismatch: "
            + ", ".join(
                f"{tracker}={cycle}"
                for tracker, cycle in latest_cycles.items()
            )
        )

    latest_cycle = next(iter(latest_cycles.values()))
    recent_start_cycle = max(1, latest_cycle - 2)

    if end < start:
        raise ValueError("--end must not precede --start")

    outdir = Path(args.output)
    outdir.mkdir(parents=True, exist_ok=False)

    products = {}

    for tracker, directory in (
        (args.tracker_a, args.tracker_a_dir),
        (args.tracker_b, args.tracker_b_dir),
    ):
        products[tracker] = {}

        for suffix in (
            "variability_historical",
            "variability_recent3",
            "variability_latest",
        ):
            df = load_product(
                directory,
                args.storm,
                tracker,
                suffix,
            )

            products[tracker][suffix] = select_window(
                df,
                start,
                end,
            )

    # Require the requested endpoints to exist in all three hierarchy
    # products for both trackers. Individual spread validity remains
    # governed by the frozen spread_valid flag.
    for tracker in (args.tracker_a, args.tracker_b):
        for suffix, df in products[tracker].items():
            times = set(df["valid_time"])

            if start not in times:
                raise RuntimeError(
                    f"{tracker} {suffix}: start time missing: {start}"
                )

            if end not in times:
                raise RuntimeError(
                    f"{tracker} {suffix}: end time missing: {end}"
                )

    summary_rows = []

    for tracker in (args.tracker_a, args.tracker_b):
        d = products[tracker]

        summary_rows.extend(
            build_summary(
                tracker,
                d["variability_historical"],
                d["variability_recent3"],
                d["variability_latest"],
                start,
                end,
            )
        )

    summary = pd.DataFrame(summary_rows)

    contraction_rows = []

    for tracker in (args.tracker_a, args.tracker_b):
        d = products[tracker]

        c67, c90, n = contraction(
            d["variability_historical"],
            d["variability_recent3"],
        )

        contraction_rows.append(
            {
                "tracker": tracker,
                "window_start": start,
                "window_end": end,
                "n_matched_valid_times": n,
                "mean_r67_contraction_pct": c67,
                "mean_r90_contraction_pct": c90,
            }
        )

    contractions = pd.DataFrame(contraction_rows)

    summary_path = (
        outdir
        / f"{args.storm.lower()}_tracker_matched_summary.csv"
    )
    contraction_path = (
        outdir
        / f"{args.storm.lower()}_tracker_matched_contraction.csv"
    )

    summary.to_csv(summary_path, index=False)
    contractions.to_csv(contraction_path, index=False)

    # Long-form pointwise table for reproducibility.
    pointwise = []

    for tracker in (args.tracker_a, args.tracker_b):
        for hierarchy, suffix in (
            ("historical", "variability_historical"),
            ("recent3", "variability_recent3"),
            ("latest", "variability_latest"),
        ):
            d = products[tracker][suffix].copy()
            d.insert(0, "hierarchy", hierarchy)
            d.insert(0, "tracker", tracker)
            pointwise.append(d)

    pointwise = pd.concat(pointwise, ignore_index=True)

    pointwise_path = (
        outdir
        / f"{args.storm.lower()}_tracker_matched_pointwise.csv"
    )
    pointwise.to_csv(pointwise_path, index=False)

    # Figure: historical, recent, and current-cycle r67/r90.
    fig, axes = plt.subplots(
        2,
        3,
        figsize=(18, 9),
        sharex="col",
    )

    hierarchy = [
        (
            "variability_historical",
            "Historical — all cycles",
        ),
        (
            "variability_recent3",
            f"Recent — cycles {recent_start_cycle}–{latest_cycle}",
        ),
        (
            "variability_latest",
            f"Current — cycle {latest_cycle}",
        ),
    ]

    for j, (suffix, title) in enumerate(hierarchy):
        plot_metric(
            axes[0, j],
            products[args.tracker_a][suffix],
            products[args.tracker_b][suffix],
            args.tracker_a,
            args.tracker_b,
            "r67_km",
            "r67 (km)",
            title,
        )

        plot_metric(
            axes[1, j],
            products[args.tracker_a][suffix],
            products[args.tracker_b][suffix],
            args.tracker_a,
            args.tracker_b,
            "r90_km",
            "r90 (km)",
            "",
        )

    fig.suptitle(
        f"AIWeather — {args.storm} "
        "WuDuan vs Vitart Matched-Window Variability\n"
        f"{start:%d %b %Y %H UTC} – "
        f"{end:%d %b %Y %H UTC}",
        fontsize=16,
    )

    fig.text(
        0.5,
        0.015,
        "Matched comparison window; individual curves are shown only "
        "where each tracker's frozen spread-valid criterion is satisfied. "
        "Spread describes forecast disagreement, not calibrated TC probability.",
        ha="center",
        fontsize=9,
    )

    fig.tight_layout(rect=(0, 0.05, 1, 0.93))

    figure_path = (
        outdir
        / f"{args.storm.lower()}_cycle{latest_cycle}_"
          "wuduan_vitart_matched_comparison.png"
    )

    fig.savefig(
        figure_path,
        dpi=200,
        bbox_inches="tight",
    )
    plt.close(fig)

    print()
    print("=" * 78)
    print(
        f" AIWEATHER MATCHED TRACKER COMPARISON — "
        f"{args.storm.upper()}"
    )
    print("=" * 78)

    print()
    print("Matched window:")
    print(f"  {start} -> {end}")

    print()
    print("MEAN SPREAD OVER MATCHED WINDOW")
    print(
        summary.to_string(
            index=False,
            float_format=lambda x: f"{x:.1f}",
        )
    )

    print()
    print("HISTORICAL -> RECENT CONTRACTION")
    print(
        contractions.to_string(
            index=False,
            float_format=lambda x: f"{x:.1f}",
        )
    )

    print()
    print("Outputs:")
    print(summary_path)
    print(contraction_path)
    print(pointwise_path)
    print(figure_path)

    print()
    print(
        "NOTE: comparison uses identical valid times and each "
        "tracker's frozen spread_valid definition."
    )


if __name__ == "__main__":
    main()
