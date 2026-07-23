"""
Create a small ERA5-like NetCDF dataset for AIWeather tests.

The generated dataset is intentionally tiny so it can be committed
to the repository and used in automated tests.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr


def main() -> None:
    """Generate a small synthetic ERA5 dataset."""

    output = Path("tests/data/era5_sample.nc")
    output.parent.mkdir(parents=True, exist_ok=True)

    time = pd.date_range("2026-01-01", periods=2, freq="6h")

    latitude = np.array([30.0, 29.75, 29.5])

    longitude = np.array([240.0, 240.25, 240.5, 240.75])

    shape = (len(time), len(latitude), len(longitude))

    rng = np.random.default_rng(seed=42)

    ds = xr.Dataset(
        data_vars={
            "t2m": (
                ("time", "latitude", "longitude"),
                273.15 + 15 + rng.random(shape),
            ),
            "u10": (
                ("time", "latitude", "longitude"),
                rng.normal(5.0, 1.0, shape),
            ),
            "v10": (
                ("time", "latitude", "longitude"),
                rng.normal(0.0, 1.0, shape),
            ),
        },
        coords={
            "time": time,
            "latitude": latitude,
            "longitude": longitude,
        },
        attrs={
            "title": "AIWeather synthetic ERA5 dataset",
            "institution": "AIWeather",
            "Conventions": "CF-1.10",
        },
    )

    ds.to_netcdf(output)

    print(f"Dataset written to {output}")

    print(ds)


if __name__ == "__main__":
    main()