"""
Storm-center loaders for the AIFS2 tropical-cyclone wave workflow.

All supported center sources are normalized to the same columns:

    lead_time_hours
    center_latitude
    center_longitude
    center_longitude_plot

Supported sources
-----------------
wuduan
    WuDuan tracker records from an AIWeather batch_points.csv file.

native
    Standalone AIWeather native tracker CSV.

ibtracs
    Standalone IBTrACS reference CSV produced by AIWeather verification.

operational
    Operational existing-TC track CSV exported by AIWeather.
"""

from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_WUDUAN_PATH = Path(
    "results/verification/batch/"
    "aifs2_epac_4storm/"
    "batch_points.csv"
)


def to_lon180(values):
    """
    Convert longitude to the -180 to 180 convention.
    """

    values = np.asarray(
        values,
        dtype=float,
    )

    return (
        (values + 180.0) % 360.0
        - 180.0
    )


def load_wave_centers(case):
    """
    Load and normalize storm centers for one wave-analysis case.
    """

    source = case.get(
        "center_source",
        "wuduan",
    ).lower()

    case_id = case[
        "case_id"
    ]

    if source == "wuduan":

        path = Path(
            case.get(
                "center_path",
                DEFAULT_WUDUAN_PATH,
            )
        )

        df = pd.read_csv(
            path
        )

        required = {
            "case_id",
            "tracker",
            "coverage",
            "lead_time_hours",
            "forecast_latitude",
            "forecast_longitude",
        }

        missing = required - set(
            df.columns
        )

        if missing:
            raise ValueError(
                f"WuDuan center file {path} "
                f"is missing columns: "
                f"{sorted(missing)}"
            )

        centers = (
            df[
                (
                    df["case_id"]
                    == case_id
                )
                & (
                    df["tracker"]
                    == "wuduan"
                )
                & (
                    df["coverage"]
                    == "full"
                )
            ]
            .loc[
                :,
                [
                    "lead_time_hours",
                    "forecast_latitude",
                    "forecast_longitude",
                ],
            ]
            .rename(
                columns={
                    "forecast_latitude":
                        "center_latitude",
                    "forecast_longitude":
                        "center_longitude",
                }
            )
            .copy()
        )

    elif source in {
        "native",
        "ibtracs",
        "operational",
    }:

        if "center_path" not in case:
            raise ValueError(
                f"Case {case_id!r} uses "
                f"center_source={source!r} "
                "but has no center_path."
            )

        path = Path(
            case[
                "center_path"
            ]
        )

        df = pd.read_csv(
            path
        )

        required = {
            "lead_time_hours",
            "latitude",
            "longitude",
        }

        missing = required - set(
            df.columns
        )

        if missing:
            raise ValueError(
                f"{source} center file {path} "
                f"is missing columns: "
                f"{sorted(missing)}"
            )

        centers = (
            df.loc[
                :,
                [
                    "lead_time_hours",
                    "latitude",
                    "longitude",
                ],
            ]
            .rename(
                columns={
                    "latitude":
                        "center_latitude",
                    "longitude":
                        "center_longitude",
                }
            )
            .copy()
        )

    else:
        raise ValueError(
            f"Unsupported wave center source "
            f"{source!r}. "
            "Expected one of: "
            "wuduan, native, ibtracs, operational."
        )

    centers = (
        centers
        .dropna(
            subset=[
                "lead_time_hours",
                "center_latitude",
                "center_longitude",
            ]
        )
        .sort_values(
            "lead_time_hours"
        )
        .drop_duplicates(
            subset=[
                "lead_time_hours",
            ],
            keep="first",
        )
        .reset_index(
            drop=True
        )
    )

    if centers.empty:
        raise RuntimeError(
            f"No {source} center records "
            f"found for case {case_id!r}."
        )

    centers[
        "center_longitude_plot"
    ] = to_lon180(
        centers[
            "center_longitude"
        ].values
    )

    centers[
        "center_source"
    ] = source

    return centers
