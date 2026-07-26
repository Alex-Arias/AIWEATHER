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

    def preprocess(self, dataset: xr.Dataset) -> xr.Dataset:
        """
        Placeholder implementation.

        Later this will:
        - validate variables
        - interpolate grids
        - normalize data
        - build GraphCast tensors
        """
        return dataset