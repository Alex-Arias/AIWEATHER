"""
Model registry for AIWeather.
"""

from __future__ import annotations

from typing import Callable

from .base_model import BaseModel


_MODEL_REGISTRY: dict[str, Callable[..., BaseModel]] = {}


def register_model(
    name: str,
    model_class: Callable[..., BaseModel],
) -> None:
    """Register a model."""

    key = name.lower()

    if key in _MODEL_REGISTRY:
        raise ValueError(
            f"Model '{name}' is already registered."
        )

    _MODEL_REGISTRY[key] = model_class


def load_model(
    name: str,
    *args,
    **kwargs,
) -> BaseModel:
    """Instantiate a registered model."""

    key = name.lower()

    if key not in _MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model '{name}'."
        )

    return _MODEL_REGISTRY[key](*args, **kwargs)


def available_models() -> list[str]:
    """Return registered models."""

    return sorted(_MODEL_REGISTRY.keys())