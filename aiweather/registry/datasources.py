"""
Datasource registry.

Maps AIWeather prognostic models to the Earth2Studio
datasource appropriate for their required input fields.
"""

from __future__ import annotations

from earth2studio.data import (
    GFS,
    IFS,
)


_DATASOURCE_REGISTRY = {
    "graphcast": {
        "name": "gfs",
        "class": GFS,
    },
    "aifs2": {
        "name": "ifs",
        "class": IFS,
    },
}


def get_data_source(
    model_name: str,
):
    """
    Return the datasource configuration associated with a model.
    """

    if not isinstance(
        model_name,
        str,
    ):
        raise TypeError(
            "model_name must be a string."
        )

    key = model_name.lower()

    try:
        return _DATASOURCE_REGISTRY[
            key
        ]

    except KeyError:
        raise ValueError(
            f"No datasource is configured "
            f"for model '{model_name}'. "
            f"Configured models: "
            f"{', '.join(sorted(_DATASOURCE_REGISTRY))}"
        )