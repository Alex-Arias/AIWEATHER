"""
Earth2Studio model backend.
"""

from __future__ import annotations

from aiweather.registry.models import get_px_model


def load_px_model(model_name: str):
    """
    Load a PX model from the AIWeather registry.
    """

    ModelClass = get_px_model(model_name)

    package = ModelClass.load_default_package()

    return ModelClass.load_model(package)