"""
Custom exceptions used throughout AIWeather.
"""

from __future__ import annotations


class AIWeatherError(Exception):
    """
    Base exception for all AIWeather errors.
    """


class ValidationError(AIWeatherError):
    """
    Raised when an input dataset fails validation.
    """


class ModelNotLoadedError(AIWeatherError):
    """
    Raised when inference is attempted before a model is loaded.
    """


class UnsupportedVariableError(AIWeatherError):
    """
    Raised when unsupported variables are requested.
    """