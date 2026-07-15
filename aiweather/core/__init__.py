"""
Core classes and utilities for AIWeather.

The core package contains common enumerations, exceptions, metadata,
configuration classes, and other foundational components used
throughout the framework.
"""


from .configuration import Configuration
from .exceptions import (
    AIWeatherError,
    CacheError,
    ConfigurationError,
    DatasetError,
    DownloadError,
    ExportError,
    ForecastError,
    HPCError,
    ModelError,
    VerificationError,
)
from .states import ForecastStatus

__all__ = [
    "Configuration",
    "ForecastStatus",
    "AIWeatherError",
    "ConfigurationError",
    "DatasetError",
    "DownloadError",
    "CacheError",
    "ModelError",
    "ForecastError",
    "VerificationError",
    "ExportError",
    "HPCError",
]