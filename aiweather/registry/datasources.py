"""
Datasource registry.

Currently every AIWeather prognostic model uses GFS.

Future versions may map individual models to ERA5, HRRR,
IFS, custom datasets, etc.
"""

from __future__ import annotations

from earth2studio.data import GFS


DEFAULT_DATASOURCE = GFS


def get_data_source(model_name: str):
    """
    Return the datasource class associated with a model.
    """

    return DEFAULT_DATASOURCE