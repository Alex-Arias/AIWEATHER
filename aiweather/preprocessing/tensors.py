"""
Tensor conversion utilities.
"""

from __future__ import annotations

import numpy as np
import torch
import xarray as xr


def dataset_to_tensor(dataset: xr.Dataset) -> torch.Tensor:
    """
    Convert an xarray Dataset into a tensor.

    Output dimensions:
        (time, variable, lat, lon)
    """

    arrays = np.stack(
        [dataset[var].values for var in dataset.data_vars],
        axis=0,
    )

    tensor = torch.tensor(arrays, dtype=torch.float32)

    tensor = tensor.permute(1, 0, 2, 3)

    return tensor