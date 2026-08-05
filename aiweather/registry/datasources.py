"""
Registry of data sources used by PX models.
"""

from __future__ import annotations

from earth2studio.data import GFS

DATA_REGISTRY = {
    "graphcast": GFS,
}


def get_data_source(name: str):
    """
    Return the datasource class associated with a model.
    """
    try:
        return DATA_REGISTRY[name]
    except KeyError:
        raise ValueError(
            f"Unknown datasource for model '{name}'. "
            f"Available models: {list(DATA_REGISTRY.keys())}"
        )