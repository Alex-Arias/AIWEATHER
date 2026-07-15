"""
AIWeather Core Package.

This package contains the fundamental classes and utilities shared
by every AIWeather forecasting model and workflow.

As the framework evolves, this package will provide the common API
used by GraphCast, AIFS, AIFS2, Aurora, FengWu, StormCast, Pangu,
and future AI weather prediction models.
"""

from .states import ForecastStatus

__all__ = [
    "ForecastStatus",
]