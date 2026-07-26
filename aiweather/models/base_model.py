"""
Abstract base class for AIWeather forecast models.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from aiweather.core import ForecastMetadata


class BaseModel(ABC):
    """
    Abstract interface for every AIWeather forecasting model.
    """

    def __init__(
        self,
        name: str,
        version: str = "unknown",
        device: str = "cpu",
    ):
        self.name = name
        self.version = version
        self.device = device
        self._loaded = False

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}"
            f"(name={self.name!r}, "
            f"version={self.version!r}, "
            f"device={self.device!r})"
        )

    @property
    def loaded(self) -> bool:
        """
        Return whether model weights have been loaded.
        """
        return self._loaded

    @abstractmethod
    def load_weights(
        self,
        path: str | Path | None = None,
    ) -> None:
        """
        Load model weights.
        """

    @abstractmethod
    def supported_variables(self) -> tuple[str, ...]:
        """
        Return variables supported by the model.
        """

    @abstractmethod
    def run(
        self,
        dataset: Any,
        lead_time: int,
    ) -> Any:
        """
        Run the forecast.
        """

    def create_metadata(
        self,
        **kwargs,
    ) -> ForecastMetadata:
        """
        Construct ForecastMetadata for this forecast.
        """

        return ForecastMetadata(
            model_name=self.name,
            **kwargs,
        )