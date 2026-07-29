"""
GraphCast adapter.
"""

from __future__ import annotations

from aiweather.postprocessing import tensor_to_dataset

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

        print()
        print("Generator created")
        print()

        print("Getting first forecast...")
        print()

        forecast_tensor, forecast_coords = next(iterator)

        print("Forecast tensor:")
        print(type(forecast_tensor))

        if hasattr(forecast_tensor, "shape"):
            print(forecast_tensor.shape)

        print()

        print("Forecast coords:")
        print(type(forecast_coords))
        print(forecast_coords)

        return forecast_tensor, forecast_coords


#        iterator = self.model.create_iterator(
#            tensor,
#            coords,
#        )
#
#        return iterator
