"""
Automatic registry of all supported Earth2Studio PX models.

AIWeather discovers every prognostic model available in the installed
Earth2Studio version and exposes them through a stable public API.
"""

from __future__ import annotations

import inspect

import earth2studio.models.px as px


# ---------------------------------------------------------------------
# Internal classes that should never appear as forecast models
# ---------------------------------------------------------------------

_SKIP = {
    "PrognosticModel",
    "DiagnosticWrapper",
    "DataReplay",
}


# ---------------------------------------------------------------------
# Discover every model class automatically
# ---------------------------------------------------------------------

_MODEL_REGISTRY = {}

for name, obj in inspect.getmembers(px):

    if not inspect.isclass(obj):
        continue

    if name.startswith("_"):
        continue

    if name in _SKIP:
        continue

    _MODEL_REGISTRY[name.lower()] = obj


# ---------------------------------------------------------------------
# Friendly names exposed by AIWeather
# ---------------------------------------------------------------------

ALIASES = {
    "graphcast": "graphcastoperational",
    "graphcast_small": "graphcastsmall",
    "gencast": "gencastmini",
    "atlas": "atlas",
    "aurora": "aurora",
    "aurora1p5": "aurora1p5",
    "stormcast": "stormcast",
    "stormcastconus": "stormcastconus",
    "stormscopegoes": "stormscopegoes",
    "stormscopemrms": "stormscopemrms",
    "pangu24": "pangu24",
    "pangu3": "pangu3",
    "pangu6": "pangu6",
    "aifs": "aifs",
    "aifs2": "aifs2",
    "aifs2ens": "aifs2ens",
    "aifsens": "aifsens",
    "fengwu": "fengwu",
    "fuxi": "fuxi",
    "fcn": "fcn",
    "fcn3": "fcn3",
    "sfno": "sfno",
    "dlwp": "dlwp",
    "dlesym": "dlesym",
    "dlesymlatlon": "dlesymlatlon",
    "interpmodafno": "interpmodafno",
    "ucast": "ucast",
    "ace2": "ace2era5",
    "persistence": "persistence",
}


# ---------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------

def available_models():
    """Return all AIWeather-supported model names."""

    return sorted(ALIASES.keys())


def get_px_model(name: str):

    key = name.lower()

    if key in ALIASES:
        key = ALIASES[key]

    try:
        return _MODEL_REGISTRY[key]

    except KeyError:

        raise ValueError(
            f"Unknown model '{name}'.\n"
            f"Available models:\n"
            f"{', '.join(available_models())}"
        )