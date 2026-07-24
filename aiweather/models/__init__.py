"""
AIWeather model interfaces.
"""

from .base_model import BaseModel
from .registry import (
    register_model,
    load_model,
    available_models,
)

__all__ = [
    "BaseModel",
    "register_model",
    "load_model",
    "available_models",
]