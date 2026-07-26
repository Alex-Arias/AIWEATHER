"""
Base class for AIWeather preprocessing pipelines.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import xarray as xr


class BasePreprocessor(ABC):
    """
    Abstract preprocessing interface.
    """

    @abstractmethod
    def preprocess(self, dataset: xr.Dataset) -> xr.Dataset:
        """
        Convert an input dataset into the format required by a model.
        """