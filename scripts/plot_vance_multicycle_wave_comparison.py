#!/usr/bin/env python3
"""QC-aware AIFS2 wave comparison for Vance Cycles 1–5.

Uses existing wave diagnostics only. No forecast reruns.
Successive cycles are compared at matching valid times.

Cycle 4: retain leads 0–150 h.
Cycle 5: retain leads 0–126 h.
"""

import hashlib
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path("results/waves")

OUTPUT_DIR = ROOT / "vance_cycles1_5_comparison"

OUTPUT_PNG = (
    OUTPUT_DIR
    / "vance_cycles1_5_wave_comparison_QC_VALIDATED.png"
)

OUTPUT_PDF = OUTPUT_PNG.with_suffix(".pdf")

CASES = {
    1: (
        "20261021T060000",
        ROOT / "aifs2_invest_94e_20261021"
        / "aifs2_invest_94e_wave_diagnostics.csv",
        228,
    ),
    2: (
        "20261022T060000",
        ROOT / "aifs2_invest_94e_20261022"
        / "aifs2_invest_94e_wave_diagnostics.csv",
        180,
    ),
    3: (
        "20261023T060000",
        ROOT / "aifs2_invest_94e_20261023"
        / "aifs2_vance_wave_diagnostics.csv",
        186,
    ),
    4: (
        "20261024T060000",
        ROOT / "aifs2_invest_94e_20261024"
        / "aifs2_vance_wave_diagnostics.csv",
        150,
    ),
    5: (
        "20261025T060000",
        ROOT / "aifs2_invest_94e_20261025"
        / "aifs2_vance_wave_diagnostics.csv",
        126,
    ),
}

REQUIRED = {
    "lead_time_hours",
    "max_swh_300km_m",
    "max_wind_300km_ms",
}

STYLES = ["-", "--", ":", "-.", "-"]

COLORS = [
    "tab:blue",
    "tab:orange",
    "tab:green",
    "tab:red",
    "tab:purple",
]


def load_cases():

    data = {}

    for cycle, (init, path, max_lead) in CASES.items():

        if not path.is_file():
            raise FileNotFoundError(path)

        df = pd.read_csv(path)

        if not REQUIRED.issubset(df.columns):
            raise ValueError(
                f"Invalid diagnostics schema: Cycle {cycle}"
            )

        df = df.loc[
            df["lead_time_hours"] <= max_lead
        ].copy()

        df = df.sort_values(
            "lead_time_hours"
        )

        if df.empty:
            raise ValueError(
                f"No accepted records: Cycle {cycle}"
            )

        if df["lead_time_hours"].duplicated().any():
            raise ValueError(
                f"Duplicate forecast leads: Cycle {cycle}"
            )

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

        data[cycle] = df

    return data


def make_figure(data):

    fig, axes = plt.subplots(
        3,
        1,
        figsize=(11, 15),
        sharex=True,
        constrained_layout=True,
    )

    ax_swh, ax_wind, ax_delta = axes

    for cycle, df in data.items():

        init, _, _ = CASES[cycle]

        label = pd.to_datetime(
            init,
            format="%Y%m%dT%H%M%S",
        ).strftime("%Y-%m-%d %H UTC")

        style = STYLES[cycle - 1]
        color = COLORS[cycle - 1]

        ax_swh.plot(
            df["valid_time"],
            df["max_swh_300km_m"],
            linestyle=style,
            color=color,
            marker="o",
            markersize=3,
            linewidth=2,
            label=label,
        )

        ax_wind.plot(
            df["valid_time"],
            df["max_wind_300km_ms"],
            linestyle=style,
            color=color,
            marker="o",
            markersize=3,
            linewidth=2,
            label=label,
        )

    for old_cycle in range(1, 5):

        new_cycle = old_cycle + 1

        old = data[old_cycle][
            ["valid_time", "max_swh_300km_m"]
        ]

        new = data[new_cycle][
            ["valid_time", "max_swh_300km_m"]
        ]

        common = old.merge(
            new,
            on="valid_time",
            suffixes=("_old", "_new"),
            validate="one_to_one",
        )

        if common.empty:
            raise ValueError(
                f"No overlapping times: "
                f"Cycles {old_cycle}–{new_cycle}"
            )

        common["delta_swh_m"] = (
            common["max_swh_300km_m_new"]
            - common["max_swh_300km_m_old"]
        )

        color = COLORS[old_cycle - 1]
        style = STYLES[old_cycle - 1]

        label = (
            f"{20 + new_cycle} Oct - "
            f"{20 + old_cycle} Oct"
        )

        ax_delta.plot(
            common["valid_time"],
            common["delta_swh_m"],
            color=color,
            linestyle=style,
            linewidth=2,
            marker="o",
            markersize=3,
            label=label,
        )

        peak_index = (
            common["delta_swh_m"]
            .abs()
            .idxmax()
        )

        peak = common.loc[peak_index]

        ax_delta.scatter(
            peak["valid_time"],
            peak["delta_swh_m"],
            color=color,
            s=55,
            zorder=5,
        )

        ax_delta.annotate(
            f"{peak['delta_swh_m']:+.2f} m",
            xy=(
                peak["valid_time"],
                peak["delta_swh_m"],
            ),
            xytext=(8, 8),
            textcoords="offset points",
            fontsize=9,
            color=color,
        )

        print(
            f"Cycles {old_cycle} → {new_cycle}: "
            f"{len(common)} common times; "
            f"peak revision {peak['delta_swh_m']:+.3f} m"
        )

    ax_delta.axhline(
        0,
        color="black",
        linestyle=":",
        linewidth=1,
    )

    ax_swh.set_title("Vance")

    ax_swh.set_ylabel(
        "Maximum SWH within 300 km (m)"
    )

    ax_wind.set_ylabel(
        "Maximum wind within 300 km (m s$^{-1}$)"
    )

    ax_delta.set_ylabel(
        "$\\Delta$SWH between successive cycles (m)"
    )

    ax_delta.set_xlabel(
        "Forecast valid time (UTC)"
    )

    for ax in axes:

        ax.grid(True, alpha=0.3)

        ax.xaxis.set_major_locator(
            mdates.DayLocator(interval=1)
        )

        ax.xaxis.set_major_formatter(
            mdates.DateFormatter(
                "%d %b",
                tz=mdates.UTC,
            )
        )

    ax_swh.legend(
        title="Initialization",
        loc="upper right",
    )

    ax_delta.legend(
        title="Cycle revision",
        fontsize=9,
        loc="upper right",
    )

    fig.suptitle(
        "AIFS2 Operational Wave Forecast Cycle Comparison\n"
        "Vance — 21–25 October 2026",
        fontsize=15,
    )

    return fig


def main():

    for path in (OUTPUT_PNG, OUTPUT_PDF):
        if path.exists():
            raise SystemExit(
                f"STOP — Existing product: {path}"
            )

    data = load_cases()

    print("=" * 72)
    print(" VANCE — FIVE-CYCLE WAVE COMPARISON")
    print("=" * 72)

    for cycle, df in data.items():

        print(
            f"Cycle {cycle}: "
            f"{len(df)} accepted records, "
            f"+{df['lead_time_hours'].min()} "
            f"to +{df['lead_time_hours'].max()} h"
        )

    fig = make_figure(data)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    fig.savefig(
        OUTPUT_PNG,
        dpi=250,
        bbox_inches="tight",
    )

    fig.savefig(
        OUTPUT_PDF,
        bbox_inches="tight",
    )

    plt.close(fig)

    print("\n---- GENERATED PRODUCTS ----")

    for path in (OUTPUT_PNG, OUTPUT_PDF):

        print(
            path,
            path.stat().st_size,
            "bytes",
            hashlib.sha256(
                path.read_bytes()
            ).hexdigest(),
        )

    print("\nPASS — Five-cycle comparison generated")


if __name__ == "__main__":
    main()
