"""
Base runner for AIWeather forecast engines.
"""

from __future__ import annotations

from abc import ABC, abstractmethod


class BaseRunner(ABC):
    """
    Base class for all operational forecast runners.
    """

    @abstractmethod
    def run(
        self,
        request,
    ):
        """
        Execute a forecast.
        """
        raise NotImplementedError