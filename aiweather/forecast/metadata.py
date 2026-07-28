"""
Forecast metadata.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime


@dataclass(slots=True)
class ForecastMetadata:
    """
    Metadata describing a forecast.
    """

    model_name: str
    model_version: str
    backend: str

    forecast_id: str | None = None

    initialization_time: datetime | None = None

    creation_time: datetime | None = None