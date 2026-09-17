"""Shared plotting styles for AIWeather forecast models."""

from __future__ import annotations


MODEL_ORDER = (
    "graphcast",
    "aifs2",
    "pangu3",
    "pangu6",
)

MODEL_LABELS = {
    "graphcast": "GraphCast",
    "aifs2": "AIFS2",
    "pangu3": "Pangu3",
    "pangu6": "Pangu6",
}

MODEL_COLORS = {
    "graphcast": "C0",
    "aifs2": "C1",
    "pangu3": "C2",
    "pangu6": "C3",
}

MODEL_MARKERS = {
    "graphcast": "s",
    "aifs2": "^",
    "pangu3": "D",
    "pangu6": "o",
}

MODEL_LINESTYLES = {
    "graphcast": "-",
    "aifs2": "-",
    "pangu3": "-",
    "pangu6": "--",
}


def model_label(model: str) -> str:
    """Return the display label for a model."""
    return MODEL_LABELS.get(
        model,
        model,
    )


def model_color(model: str) -> str:
    """Return the canonical plotting color for a model."""
    return MODEL_COLORS.get(
        model,
        "C4",
    )


def model_marker(model: str) -> str:
    """Return the canonical plotting marker for a model."""
    return MODEL_MARKERS.get(
        model,
        "o",
    )


def model_linestyle(model: str) -> str:
    """Return the canonical plotting line style for a model."""
    return MODEL_LINESTYLES.get(
        model,
        "-",
    )


def model_sort_key(
    model: str,
) -> tuple[int, str]:
    """Return a stable sort key using canonical model order."""
    try:
        index = MODEL_ORDER.index(
            model
        )
    except ValueError:
        index = len(
            MODEL_ORDER
        )

    return index, model
