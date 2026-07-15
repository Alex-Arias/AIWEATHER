"""
Identifier utilities for AIWeather.

This module provides functions for generating unique identifiers
used throughout the AIWeather framework.
"""

from __future__ import annotations

from datetime import UTC, datetime
from uuid import uuid4


def generate_forecast_id(
    prefix: str | None = None,
) -> str:
    """
    Generate a unique forecast identifier.

    Parameters
    ----------
    prefix
        Optional identifier prefix (e.g. "GC", "AIFS", "PANGU").

    Returns
    -------
    str
        Forecast identifier.

    Examples
    --------
    >>> generate_forecast_id()
    '20260715T184512Z-3f8c1e7a'

    >>> generate_forecast_id(prefix="GC")
    'GC-20260715T184512Z-3f8c1e7a'
    """

    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    unique = uuid4().hex[:8]

    identifier = f"{timestamp}-{unique}"

    if prefix:
        identifier = f"{prefix.upper()}-{identifier}"

    return identifier