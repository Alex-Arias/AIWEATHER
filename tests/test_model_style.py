from aiweather.plotting.model_style import (
    MODEL_ORDER,
    model_color,
    model_label,
    model_linestyle,
    model_marker,
    model_sort_key,
)


def test_canonical_model_styles():
    assert MODEL_ORDER == (
        "graphcast",
        "aifs2",
        "pangu3",
        "pangu6",
    )

    expected = {
        "graphcast": (
            "GraphCast",
            "C0",
            "s",
            "-",
        ),
        "aifs2": (
            "AIFS2",
            "C1",
            "^",
            "-",
        ),
        "pangu3": (
            "Pangu3",
            "C2",
            "D",
            "-",
        ),
        "pangu6": (
            "Pangu6",
            "C3",
            "o",
            "--",
        ),
    }

    for model, (
        label,
        color,
        marker,
        linestyle,
    ) in expected.items():
        assert model_label(model) == label
        assert model_color(model) == color
        assert model_marker(model) == marker
        assert model_linestyle(model) == linestyle


def test_model_sort_key_uses_canonical_order():
    models = [
        "pangu6",
        "aifs2",
        "graphcast",
        "pangu3",
    ]

    assert sorted(
        models,
        key=model_sort_key,
    ) == [
        "graphcast",
        "aifs2",
        "pangu3",
        "pangu6",
    ]


def test_unknown_model_fallbacks_are_stable():
    model = "future_model"

    assert model_label(model) == model
    assert model_marker(model) == "o"
    assert model_linestyle(model) == "-"
    assert model_sort_key(model) == (
        len(MODEL_ORDER),
        model,
    )

    # Unknown models receive a deterministic fallback
    # color so plots remain stable across call order.
    assert model_color(model) == "C4"
