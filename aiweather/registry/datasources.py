"""
Datasource registry.

Maps each model to its required Earth2Studio datasource.
"""

from earth2studio.data import GFS


DATA_REGISTRY = {

    "graphcast": GFS,

}