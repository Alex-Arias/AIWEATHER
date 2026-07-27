"""
Earth2Studio CoordSystem utilities.
"""

from __future__ import annotations

from collections import OrderedDict

import numpy as np
import xarray as xr


def dataset_to_coordsystem(dataset: xr.Dataset) -> OrderedDict:
    """
    Convert an xarray Dataset into an Earth2Studio CoordSystem.
    """

    coords = OrderedDict()

    coords["batch"] = np.array([0])

    coords["time"] = dataset.time.values

    # Initial condition (-6 h, 0 h)
    coords["lead_time"] = np.array([-6, 0])

    coords["variable"] = np.array(
        list(dataset.data_vars)
    )

    coords["lat"] = dataset.lat.values

    coords["lon"] = dataset.lon.values

    return coords