"""
GraphCast postprocessing.
"""

from __future__ import annotations

import torch


def tensor_to_dataset(
    tensor: torch.Tensor,
    coords,
):
    """
    Convert a GraphCast output tensor into an xarray.Dataset.

    This is currently a diagnostic implementation used to
    inspect the GraphCast output before building the full
    conversion pipeline.
    """

    print()
    print("===== GraphCast Output =====")
    print()

    print("Tensor shape:")
    print(tensor.shape)

    print()

    print("Coordinates:")

    for key, value in coords.items():

        print(f"{key}: {value}")

    print()

    raise NotImplementedError(
        "GraphCast tensor-to-dataset conversion not implemented yet."
    )