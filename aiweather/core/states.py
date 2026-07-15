"""
AIWeather

Core module defining the execution states of a forecast.

The ForecastStatus enumeration describes the lifecycle of a forecast
from creation to completion, verification, export, or termination.

Author
------
Alejandro Arias

License
-------
MIT
"""

from enum import Enum


class ForecastStatus(str, Enum):
    """
    Enumeration describing the lifecycle of an AIWeather forecast.

    The execution state is updated throughout the forecast workflow
    and is used for logging, monitoring, metadata generation,
    workflow management, and error handling.
    """

    UNKNOWN = "unknown"
    CREATED = "created"
    INITIALIZED = "initialized"
    RUNNING = "running"
    COMPLETED = "completed"
    VERIFIED = "verified"
    EXPORTED = "exported"
    FAILED = "failed"
    CANCELLED = "cancelled"