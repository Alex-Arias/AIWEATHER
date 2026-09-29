#!/usr/bin/env python3
"""
Plot storm-relative diagnostics for one operational AIFS2 wave cycle.

This script reads an existing wave-diagnostics CSV only. It does not
rerun forecast inference, tropical-cyclone tracking, or wave analysis.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd


def load_diagnostics(path: Path, init: str) -> pd.DataFrame:
    """Read diagnostics and reconstruct forecast valid time."""

    df = pd.read_csv(path)

    required = {
        "lead_time_hours",
        "max_swh_300km_m",
        "mean_swh_300km_m",
        "max_wind_300km_ms",
        "mean_mwp_300km_s",
    }

    missing = sorted(required - set(df.columns))

    if missing:
        raise ValueError(
            "Diagnostics missing required columns: "
            + ", ".join(missing)
        )

    df = df.sort_values(
        "lead_time_hours"
    ).reset_index(drop=True)

    init_time = pd.to_datetime(
        init,
        format="%Y%m%dT%H%M%S",
        utc=True,
    )

    df["valid_time"] = (
        init_time
        + pd.to_timedelta(
            df["lead_time_hours"],
            unit="h",
        )
    )

    return df


def make_figure(
    df: pd.DataFrame,
    *,
    storm: str,
    init: str,
    output: Path,
) -> None:
    """Create the single-cycle operational wave-evolution figure."""

    init_time = pd.to_datetime(
        init,
        format="%Y%m%dT%H%M%S",
        utc=True,
    )

    fig, axes = plt.subplots(
        3,
        1,
        figsize=(12, 10),
        sharex=True,
        constrained_layout=True,
    )

    # --------------------------------------------------------
    # Panel 1 — SWH
    # --------------------------------------------------------

    ax = axes[0]

    ax.plot(
        df["valid_time"],
        df["max_swh_300km_m"],
        marker="o",
        markersize=3.5,
        linewidth=2.0,
        label="Maximum SWH ≤ 300 km",
    )

    ax.plot(
        df["valid_time"],
        df["mean_swh_300km_m"],
        linestyle="--",
        linewidth=1.8,
        label="Mean SWH ≤ 300 km",
    )

    peak_swh_index = (
        df["max_swh_300km_m"].idxmax()
    )

    peak_swh = df.loc[peak_swh_index]

    ax.scatter(
        peak_swh["valid_time"],
        peak_swh["max_swh_300km_m"],
        s=55,
        zorder=5,
    )

    ax.annotate(
        (
            f"{peak_swh['max_swh_300km_m']:.2f} m\n"
            f"+{int(peak_swh['lead_time_hours'])} h"
        ),
        (
            peak_swh["valid_time"],
            peak_swh["max_swh_300km_m"],
        ),
        xytext=(10, -10),
        textcoords="offset points",
        fontsize=9,
        ha="left",
        va="top",
    )

    ax.set_ylabel("SWH (m)")
    ax.set_title(
        "Storm-relative significant wave height"
    )
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)

    # --------------------------------------------------------
    # Panel 2 — wind
    # --------------------------------------------------------

    ax = axes[1]

    ax.plot(
        df["valid_time"],
        df["max_wind_300km_ms"],
        marker="o",
        markersize=3.5,
        linewidth=2.0,
        label="Maximum 10-m wind ≤ 300 km",
    )

    peak_wind_index = (
        df["max_wind_300km_ms"].idxmax()
    )

    peak_wind = df.loc[peak_wind_index]

    ax.scatter(
        peak_wind["valid_time"],
        peak_wind["max_wind_300km_ms"],
        s=55,
        zorder=5,
    )

    ax.annotate(
        (
            f"{peak_wind['max_wind_300km_ms']:.2f} m s$^{{-1}}$\n"
            f"+{int(peak_wind['lead_time_hours'])} h"
        ),
        (
            peak_wind["valid_time"],
            peak_wind["max_wind_300km_ms"],
        ),
        xytext=(10, -10),
        textcoords="offset points",
        fontsize=9,
        ha="left",
        va="top",
    )

    ax.set_ylabel(
        "Wind speed (m s$^{-1}$)"
    )
    ax.set_title(
        "Storm-relative wind forcing"
    )
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)

    # --------------------------------------------------------
    # Panel 3 — wave period
    # --------------------------------------------------------

    ax = axes[2]

    ax.plot(
        df["valid_time"],
        df["mean_mwp_300km_s"],
        marker="o",
        markersize=3.5,
        linewidth=2.0,
        label="Mean wave period ≤ 300 km",
    )

    ax.set_ylabel("Mean wave period (s)")
    ax.set_xlabel("Forecast valid time (UTC)")
    ax.set_title(
        "Storm-relative mean wave period"
    )
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)

    # --------------------------------------------------------
    # Time formatting
    # --------------------------------------------------------

    locator = mdates.DayLocator()
    formatter = mdates.DateFormatter(
        "%d %b",
        tz=mdates.UTC,
    )

    for ax in axes:
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(formatter)

    fig.suptitle(
        (
            f"AIFS2 Operational Wave Evolution — {storm}\n"
            f"Initialization: "
            f"{init_time.strftime('%d %B %Y %H UTC')}"
        ),
        fontsize=15,
    )

    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fig.savefig(
        output,
        dpi=300,
        bbox_inches="tight",
    )

    pdf_output = output.with_suffix(".pdf")

    fig.savefig(
        pdf_output,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(f"Created: {output}")
    print(f"Created: {pdf_output}")


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Plot one operational AIFS2 storm-relative "
            "wave forecast cycle."
        )
    )

    parser.add_argument(
        "--storm",
        required=True,
    )

    parser.add_argument(
        "--init",
        required=True,
        help="Initialization as YYYYMMDDTHHMMSS.",
    )

    parser.add_argument(
        "--diagnostics",
        required=True,
        type=Path,
    )

    parser.add_argument(
        "--output",
        required=True,
        type=Path,
    )

    args = parser.parse_args()

    df = load_diagnostics(
        args.diagnostics,
        args.init,
    )

    make_figure(
        df,
        storm=args.storm,
        init=args.init,
        output=args.output,
    )

    peak_swh = df.loc[
        df["max_swh_300km_m"].idxmax()
    ]

    peak_wind = df.loc[
        df["max_wind_300km_ms"].idxmax()
    ]

    print()
    print("=" * 72)
    print("OPERATIONAL WAVE EVOLUTION SUMMARY")
    print("=" * 72)
    print(
        "Maximum 300-km SWH : "
        f"{peak_swh['max_swh_300km_m']:.2f} m "
        f"at +{int(peak_swh['lead_time_hours'])} h"
    )
    print(
        "Maximum 300-km wind: "
        f"{peak_wind['max_wind_300km_ms']:.2f} m/s "
        f"at +{int(peak_wind['lead_time_hours'])} h"
    )
    print(
        "Forecast window     : "
        f"{df['valid_time'].min()} -> "
        f"{df['valid_time'].max()}"
    )


if __name__ == "__main__":
    main()
