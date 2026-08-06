"""
AIWeather registry.
"""

from .models import (
    available_models,
    get_px_model,
)

from .datasources import get_data_source

__all__ = [
    "available_models",
    "get_px_model",
    "get_data_source",
]