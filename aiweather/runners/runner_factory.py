"""
Runner factory.
"""

from __future__ import annotations

from .graphcast_runner import GraphCastRunner


def get_runner(model_name: str):

    model = model_name.lower()

    if model == "graphcast":
        return GraphCastRunner()

    raise ValueError(
        f"Unknown model '{model_name}'."
    )