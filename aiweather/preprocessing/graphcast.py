"""
GraphCast preprocessing.
"""

from __future__ import annotations

import xarray as xr

from .base import BasePreprocessor


class GraphCastPreprocessor(BasePreprocessor):
    """
    Preprocessing pipeline for GraphCast.
    """

    REQUIRED_VARIABLES = (
        "t2m",
        "msl",
        "u10m",
        "v10m",
    )

    def validate(self, dataset: xr.Dataset) -> xr.Dataset:
        """
        Validate a GraphCast input dataset.
        """

        missing = [
            var
            for var in self.REQUIRED_VARIABLES
            if var not in dataset.data_vars
        ]

        if missing:
            raise ValueError(
                f"Missing required variables: {missing}"
            )

        required_coords = (
            "time",
            "lat",
            "lon",
        )

        missing = [
            coord
            for coord in required_coords
            if coord not in dataset.coords
        ]

        if missing:
            raise ValueError(
                f"Missing coordinates: {missing}"
            )

        return dataset

    def preprocess(self, dataset: xr.Dataset) -> xr.Dataset:
        """
        Run the GraphCast preprocessing pipeline.
        """

        dataset = self.validate(dataset)

        # Future preprocessing steps:
        # dataset = self.sort_latitude(dataset)
        # dataset = self.convert_longitudes(dataset)
        # dataset = self.interpolate(dataset)
        # dataset = self.normalize(dataset)

        return dataset