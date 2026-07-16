"""
Abstract dataset interface for AIWeather.

Every dataset implementation (ERA5, GFS, IFS, etc.) must inherit from
this base class and implement its interface.

This abstraction allows forecasting models to operate independently
of the underlying data source.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any


class Dataset(ABC):
    """
    Abstract base class representing a meteorological dataset.
    """

    def __init__(self, path: str | Path):
        """
        Initialize the dataset.

        Parameters
        ----------
        path : str | Path
            Location of the dataset.
        """
        self.path = Path(path)

    @abstractmethod
    def open(self) -> None:
        """
        Open the dataset.
        """
        raise NotImplementedError

    @abstractmethod
    def close(self) -> None:
        """
        Close the dataset.
        """
        raise NotImplementedError

    @abstractmethod
    def variables(self) -> list[str]:
        """
        Return available variable names.
        """
        raise NotImplementedError

    @abstractmethod
    def times(self) -> Any:
        """
        Return available forecast or analysis times.
        """
        raise NotImplementedError

    @abstractmethod
    def grid(self) -> Any:
        """
        Return grid information.
        """
        raise NotImplementedError

    @abstractmethod
    def subset(self, **kwargs: Any) -> Any:
        """
        Return a spatial and/or temporal subset.
        """
        raise NotImplementedError

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}"
            f"(path='{self.path}')"
        )