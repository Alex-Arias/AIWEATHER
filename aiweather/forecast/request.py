"""
Forecast request object.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class ForecastRequest:
    """
    Description of a forecast to execute.
    """

    # ---------------------------------------------------------
    # Forecast configuration
    # ---------------------------------------------------------

    model: str

    datasource: str

    init_time: str

    lead_time: int

    # ---------------------------------------------------------
    # Optional output
    # ---------------------------------------------------------

    output_path: str | None = None

    # ---------------------------------------------------------
    # Optional device
    # ---------------------------------------------------------

    device: str = "cuda"

    # ---------------------------------------------------------
    # Optional metadata
    # ---------------------------------------------------------

    description: str = ""