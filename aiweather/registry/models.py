"""
Model registry.

Maps AIWeather model names to Earth2Studio model classes.
"""

from earth2studio.models.px import GraphCastOperational


MODEL_REGISTRY = {

    "graphcast": GraphCastOperational,

}