"""
Earth2Studio datasource backend.
"""

from __future__ import annotations

from aiweather.registry.datasources import (
    get_data_source,
)


def load_data_source(
    model_name: str,
    *,
    datasource: str,
    source: str | None = None,
):
    """
    Instantiate the datasource required by a model.
    """

    config = get_data_source(
        model_name
    )

    expected = config[
        "name"
    ]

    if datasource.lower() != expected:
        raise ValueError(
            f"Model '{model_name}' requires "
            f"datasource '{expected}', "
            f"not '{datasource}'."
        )

    DataClass = config[
        "class"
    ]

    if source is None:
        return DataClass()

    return DataClass(
        source=source
    )