"""
GraphCast model adapter.
"""


from __future__ import annotations

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
    ) -> Forecast:
        raise NotImplementedError(
            "GraphCast inference not implemented yet."
        )

    def unload(self) -> None:
        self._model = None
        self._loaded = False