"""
Factory for creating AIWeather runners.
"""

from __future__ import annotations

from .graphcast_runner import GraphCastRunner
from .aifs2_runner import AIFS2Runner
from .aifs2ens_runner import AIFS2ENSRunner
from .aurora1p5_runner import Aurora1p5Runner
from .pangu3_runner import Pangu3Runner
from .pangu6_runner import Pangu6Runner
from .fengwu_runner import FengWuRunner
from .fcn3_runner import FCN3Runner


RUNNERS = {
    "graphcast": GraphCastRunner,
    "aifs2": AIFS2Runner,
    "aifs2ens": AIFS2ENSRunner,
    "aurora1p5": Aurora1p5Runner,
    "pangu3": Pangu3Runner,
    "pangu6": Pangu6Runner,
    "fengwu": FengWuRunner,
    "fcn3": FCN3Runner,
}


def create_runner(model_name: str):

    try:
        Runner = RUNNERS[model_name.lower()]
    except KeyError:

        raise ValueError(
            f"Unknown runner '{model_name}'. "
            f"Available: {list(RUNNERS.keys())}"
        )

    return Runner()