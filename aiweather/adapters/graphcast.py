"""
GraphCast adapter.
"""

from __future__ import annotations

from aiweather.preprocessing import (
    GraphCastPreprocessor,
    dataset_to_tensor,
    dataset_to_coordsystem,
)

from .base import BaseAdapter


class GraphCastAdapter(BaseAdapter):
    """
    Adapter for GraphCast.
    """

    def __init__(self, model):

        self.model = model

        self.preprocessor = GraphCastPreprocessor()

    # ---------------------------------------------------------
    # Preprocessing
    # ---------------------------------------------------------

    def prepare(self, dataset):

        dataset = self.preprocessor.preprocess(dataset)

        tensor = dataset_to_tensor(dataset)

        coords = dataset_to_coordsystem(dataset)

        return tensor, coords

    # ---------------------------------------------------------
    # Model execution
    # ---------------------------------------------------------

    def run(
        self,
        dataset,
        lead_time,
    ):

        tensor, coords = self.prepare(dataset)

        iterator = self.model.create_iterator(
            tensor,
            coords,
        )

        return iterator