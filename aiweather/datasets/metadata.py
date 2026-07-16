"""
Dataset metadata definitions.

This module defines immutable metadata associated with
meteorological datasets.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True, slots=True)
class DatasetMetadata:
    """
    Metadata describing a meteorological dataset.

    Parameters
    ----------
    name
        Dataset name.
    source
        Organization providing the dataset.
    file_format
        Storage format (NetCDF, Zarr, GRIB, etc.).
    spatial_resolution
        Horizontal grid resolution.
    temporal_resolution
        Time interval between records.
    variables
        Available meteorological variables.
    vertical_levels
        Number of vertical levels, if applicable.
    """

    name: str

    source: Optional[str] = None

    file_format: Optional[str] = None

    spatial_resolution: Optional[str] = None

    temporal_resolution: Optional[str] = None

    variables: tuple[str, ...] = ()

    vertical_levels: Optional[int] = None

    def __post_init__(self) -> None:
        """
        Validate dataset metadata.
        """

        if not self.name.strip():
            raise ValueError(
                "Dataset name cannot be empty."
            )

        if self.vertical_levels is not None:
            if self.vertical_levels <= 0:
                raise ValueError(
                    "vertical_levels must be positive."
                )