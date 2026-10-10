#!/usr/bin/env python3
"""Plot experimental Mexican Pacific coastal spectral-wave evolution."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SECTORS = (
    "Los Cabos",
    "Sinaloa",
    "Nayarit",
    "Jalisco/Nayarit",
    "Colima",
)

BANDS = (
    ("h1012", "10–12 s"),
    ("h1214", "12–14 s"),
    ("h1417", "14–17 s"),
    ("h1721", "17–21 s"),
    ("h2125", "21–25 s"),
    ("h2530", "25–30 s"),
)

COLORS = (
    "#D55E00",
    "#E69F00",
    "#009E73",
    "#0072B2",
    "#CC79A7",
    "#56B4E9",
)


def validate_input(df):
    """Check the required data and time-series completeness."""

    required = {
        "sector",
        "lead_time_hours",
        "swh_mean_m",
        "band_energy_fraction",
        "n_original_valid_cells",
        "n_spectral_valid_cells",
    }

    required.update(
        f"{band}_mean_m"
        for band, _ in BANDS
    )

    missing = required.difference(df.columns)

    if missing:
        raise ValueError(
            f"Missing columns: {sorted(missing)}"
        )

    for sector in SECTORS:

        subset = df.loc[
            df["sector"] == sector
        ].sort_values("lead_time_hours")

        if len(subset) != 41:
            raise ValueError(
                f"{sector}: expected 41 forecast records"
            )

        expected_leads = np.arange(0, 241, 6)

        if not np.array_equal(
            subset["lead_time_hours"].to_numpy(),
            expected_leads,
        ):
            raise ValueError(
                f"{sector}: incomplete 6-hour lead sequence"
            )

        columns = (
            ["swh_mean_m", "band_energy_fraction"]
            + [f"{band}_mean_m" for band, _ in BANDS]
        )

        if not np.isfinite(
            subset[columns].to_numpy(dtype=float)
        ).all():
            raise ValueError(
                f"{sector}: non-finite spectral data"
            )

        if not (
            subset["n_original_valid_cells"].to_numpy()
            == subset["n_spectral_valid_cells"].to_numpy()
        ).all():
            raise ValueError(
                f"{sector}: original and spectral masks differ"
            )


def plot_spectral_evolution(df, output_prefix):
    """Create a five-panel spectral-wave evolution figure."""

    validate_input(df)

    fig, axes = plt.subplots(
        len(SECTORS),
        1,
        figsize=(13, 14),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )

    max_hs = float(
        df.loc[
            df["sector"].isin(SECTORS),
            "swh_mean_m",
        ].max()
    )

    ymax = np.ceil((max_hs + 0.2) * 2) / 2

    for ax, sector in zip(axes, SECTORS):

        subset = (
            df.loc[df["sector"] == sector]
            .sort_values("lead_time_hours")
        )

        lead = subset["lead_time_hours"].to_numpy()

        hs = subset["swh_mean_m"].to_numpy()

        peak_index = int(np.argmax(hs))
        peak_lead = float(lead[peak_index])
        peak_hs = float(hs[peak_index])

        ax.plot(
            lead,
            hs,
            color="black",
            linewidth=2.4,
            label="Total Hs",
            zorder=5,
        )

        for (band, label), color in zip(BANDS, COLORS):

            ax.plot(
                lead,
                subset[f"{band}_mean_m"].to_numpy(),
                color=color,
                linewidth=1.5,
                label=label,
            )

        ax.axvline(
            peak_lead,
            color="0.45",
            linestyle="--",
            linewidth=1.1,
        )

        ax.plot(
            peak_lead,
            peak_hs,
            marker="o",
            color="black",
            markersize=5,
        )

        ax.text(
            0.985,
            0.94,
            f"Peak +{peak_lead:.0f} h | Hs {peak_hs:.2f} m",
            transform=ax.transAxes,
            horizontalalignment="right",
            verticalalignment="top",
            fontsize=9,
            bbox={
                "facecolor": "white",
                "edgecolor": "none",
                "alpha": 0.8,
            },
        )

        ax.set_title(
            sector,
            loc="left",
            fontweight="bold",
            fontsize=11,
        )

        ax.set_ylabel("Hs (m)")

        ax.set_xlim(0, 240)
        ax.set_ylim(0, ymax)

        ax.grid(
            True,
            alpha=0.25,
            linewidth=0.6,
        )

    axes[-1].set_xlabel(
        "Forecast lead time (hours)"
    )

    axes[-1].set_xticks(
        np.arange(0, 241, 24)
    )

    handles, labels = axes[0].get_legend_handles_labels()

    # Reserve a dedicated header above the five panels.
    # The title and legend occupy separate vertical positions.
    fig.get_layout_engine().set(
        rect=(0.0, 0.0, 1.0, 0.89)
    )

    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.925),
        ncol=7,
        frameon=False,
        fontsize=9,
    )

    fig.suptitle(
        "Mexican Pacific Coastal Spectral-Wave Evolution\n"
        "AIFS2 initialization: 28 September 2026, 12 UTC",
        fontsize=14,
        fontweight="bold",
        y=0.995,
    )

    output_prefix = Path(output_prefix)

    output_prefix.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    for extension in ("png", "pdf"):

        destination = output_prefix.with_suffix(
            f".{extension}"
        )

        fig.savefig(
            destination,
            dpi=300,
            bbox_inches="tight",
        )

        print("Written:", destination)

    plt.close(fig)


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--output-prefix",
        type=Path,
        required=True,
    )

    args = parser.parse_args()

    df = pd.read_csv(args.input)

    plot_spectral_evolution(
        df,
        args.output_prefix,
    )


if __name__ == "__main__":
    main()
