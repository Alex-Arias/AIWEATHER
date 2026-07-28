"""
GraphCast model adapter.
"""

from __future__ import annotations

from aiweather.adapters import GraphCastAdapter

from earth2studio.models.px import GraphCastOperational

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

        self.adapter = None

    def load_weights(
        self,
        path=None,
    ) -> None:
        """
        Load the GraphCast model.
        """

        self._model = GraphCastOperational.from_pretrained()

        self.adapter = GraphCastAdapter(self._model)

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
        Execute a GraphCast forecast.
        """

        if not self.loaded:
            self.load_weights()

        #
        # Delegate all preprocessing to the adapter
        #
        tensor, coords = self.adapter.prepare(dataset)

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