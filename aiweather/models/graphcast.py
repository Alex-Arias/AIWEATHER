"""
GraphCast model adapter.
"""

from __future__ import annotations

from aiweather.preprocessing import (
    GraphCastPreprocessor,
    dataset_to_tensor,
    dataset_to_coordsystem,
)

from earth2studio.models.px import GraphCastOperational

from aiweather.forecast import Forecast

from .base_model import BaseModel


class GraphCastModel(BaseModel):

    def __init__(
        self,
        device: str = "cpu",
    ):
        super().__init__(
            name="GraphCast",
            version="Earth2Studio",
            device=device,
        )

        self._model = None

    def load_weights(
        self,
        path=None,
    ) -> None:
        """
        Load the GraphCast model.
        """

        self._model = GraphCastOperational.from_pretrained()
        self._loaded = True

    def supported_variables(self) -> tuple[str, ...]:
        return (
            "t2m",
            "u10",
            "v10",
            "msl",
        )

    def run(
        self,
        dataset,
        lead_time: int,
    ):
        """
        Execute the GraphCast preprocessing pipeline.

        Forecast generation will be implemented
        in the next milestone.
        """

        if not self.loaded:
            self.load_weights()

        preprocessor = GraphCastPreprocessor()

        dataset = preprocessor.preprocess(dataset)

        tensor = dataset_to_tensor(dataset)

        coords = dataset_to_coordsystem(dataset)

        print("Tensor shape:")
        print(tensor.shape)

        print()

        print("CoordSystem keys:")
        print(coords.keys())

        
        iterator = self._model.create_iterator(
            tensor,
            coords,
        )

        print()
        print("Iterator created successfully!")

        return iterator    


    def unload(self) -> None:
        self._model = None
        self._loaded = False