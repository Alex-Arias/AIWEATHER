"""
AIWeather exception hierarchy.

This module defines all custom exceptions used throughout the AIWeather
framework. Every package component should raise one of these exceptions
instead of generic Python exceptions whenever possible.
"""

from __future__ import annotations


class AIWeatherError(Exception):
    """
    Base class for all AIWeather exceptions.
    """


class ConfigurationError(AIWeatherError):
    """
    Raised when a configuration file or parameter is invalid.
    """


class DatasetError(AIWeatherError):
    """
    Raised when a required dataset cannot be accessed or processed.
    """


class DownloadError(DatasetError):
    """
    Raised when downloading a dataset fails.
    """


class CacheError(AIWeatherError):
    """
    Raised when cached files are missing or corrupted.
    """


class ModelError(AIWeatherError):
    """
    Raised when a forecast model cannot be initialized or executed.
    """


class ForecastError(AIWeatherError):
    """
    Raised during forecast generation.
    """


class VerificationError(AIWeatherError):
    """
    Raised during forecast verification.
    """


class ExportError(AIWeatherError):
    """
    Raised while exporting forecast products.
    """


class HPCError(AIWeatherError):
    """
    Raised for scheduler, GPU, or cluster execution problems.
    """