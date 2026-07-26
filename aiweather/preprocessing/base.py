"""
Abstract base class for AIWeather preprocessors.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import xarray as xr


class BasePreprocessor(ABC):
    """
    Abstract interface for preprocessing datasets before inference.
    """

    @abstractmethod
    def validate(
        self,
        dataset: xr.Dataset,
    ) -> xr.Dataset:
        """
        Validate the input dataset.

        Parameters
        ----------
        dataset
            Input dataset.

        Returns
        -------
        xr.Dataset
            Validated dataset.
        """

    @abstractmethod
    def preprocess(
        self,
        dataset: xr.Dataset,
    ) -> xr.Dataset:
        """
        Run the preprocessing pipeline.

        Parameters
        ----------
        dataset
            Input dataset.

        Returns
        -------
        xr.Dataset
            Preprocessed dataset.
        """