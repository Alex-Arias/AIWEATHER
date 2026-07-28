"""
Base adapter interface.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import xarray as xr
import torch


class BaseAdapter(ABC):
    """
    Base class for model adapters.

    An adapter converts an AIWeather Dataset into the
    tensor + coordinate system required by a specific model.
    """

    @abstractmethod
    def prepare(
        self,
        dataset: xr.Dataset,
    ) -> tuple[torch.Tensor, dict]:
        """
        Prepare model inputs.
        """
        pass