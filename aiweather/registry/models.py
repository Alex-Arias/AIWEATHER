"""
Registry of supported prognostic (PX) models.
"""

from __future__ import annotations

from earth2studio.models.px import GraphCastOperational

MODEL_REGISTRY = {
    "graphcast": GraphCastOperational,
}


def get_px_model(name: str):
    """
    Return the PX model class associated with a model name.
    """
    try:
        return MODEL_REGISTRY[name]
    except KeyError:
        raise ValueError(
            f"Unknown PX model '{name}'. "
            f"Available models: {list(MODEL_REGISTRY.keys())}"
        )