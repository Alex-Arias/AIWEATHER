"""
Earth2Studio datasource backend.
"""

from __future__ import annotations

from aiweather.registry.datasources import get_data_source


def load_data_source(model_name: str):
    """
    Instantiate the datasource associated with a PX model.
    """

    DataClass = get_data_source(model_name)

    return DataClass()