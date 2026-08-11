"""
Wind diagnostics for AIWeather.
"""

from __future__ import annotations

import numpy as np
import xarray as xr


def wind_speed(
    u: xr.DataArray,
    v: xr.DataArray,
    *,
    name: str = "wind_speed",
) -> xr.DataArray:
    """
    Calculate horizontal wind speed from vector components.

    Parameters
    ----------
    u : xr.DataArray
        Zonal wind component.

    v : xr.DataArray
        Meridional wind component.

    name : str, default="wind_speed"
        Name assigned to the resulting DataArray.

    Returns
    -------
    xr.DataArray
        Wind speed computed as sqrt(u**2 + v**2).

    Raises
    ------
    TypeError
        If u or v is not an xarray DataArray.

    ValueError
        If u and v do not have exactly matching coordinates
        and dimensions.
    """
    if not isinstance(u, xr.DataArray):
        raise TypeError(
            f"u must be an xarray.DataArray, got {type(u).__name__}"
        )

    if not isinstance(v, xr.DataArray):
        raise TypeError(
            f"v must be an xarray.DataArray, got {type(v).__name__}"
        )

    try:
        u_aligned, v_aligned = xr.align(
            u,
            v,
            join="exact",
        )
    except ValueError as exc:
        raise ValueError(
            "u and v must have identical coordinates and dimensions."
        ) from exc

    speed = np.hypot(
        u_aligned,
        v_aligned,
    )

    speed.name = name

    # Do not invent units. Preserve units only when both source
    # components explicitly provide the same units.
    u_units = u.attrs.get("units")
    v_units = v.attrs.get("units")

    speed.attrs = {
        "long_name": "horizontal wind speed",
    }

    if (
        u_units is not None
        and v_units is not None
        and u_units == v_units
    ):
        speed.attrs["units"] = u_units

    return speed
