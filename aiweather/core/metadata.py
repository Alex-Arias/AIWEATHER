"""
Metadata classes for AIWeather.

This module defines immutable metadata objects that describe
a forecast independently of its numerical data.
"""

from __future__ import annotations

import socket

from dataclasses import dataclass, field
from datetime import UTC, datetime

from aiweather.utils import generate_forecast_id
from aiweather.version import VERSION


# ---------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class DatasetInfo:
    """
    Information describing the input dataset.
    """

    name: str
    version: str | None = None
    source: str | None = None


# ---------------------------------------------------------------------
# Domain
# ---------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class DomainInfo:
    """
    Spatial domain information.
    """

    grid: str
    resolution: str
    latitude_range: tuple[float, float]
    longitude_range: tuple[float, float]


# ---------------------------------------------------------------------
# Runtime
# ---------------------------------------------------------------------

import socket
from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class RuntimeInfo:
    """
    Runtime execution information.
    """

    device: str = "CPU"

    backend: str = "Python"

    hostname: str = field(
        default_factory=socket.gethostname
    )


# ---------------------------------------------------------------------
# Forecast Metadata
# ---------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ForecastMetadata:
    """
    Immutable metadata describing a forecast.
    """

    model_name: str

    initialization_time: datetime

    forecast_hours: int

    variables: tuple[str, ...]

    dataset: DatasetInfo

    domain: DomainInfo

    runtime: RuntimeInfo = field(default_factory=RuntimeInfo)

    forecast_id: str = field(default_factory=generate_forecast_id)

    creation_time: datetime = field(
        default_factory=lambda: datetime.now(UTC)
    )

    aiweather_version: str = VERSION