"""
Earth2Studio inference backend.
"""

from __future__ import annotations

from datetime import datetime

from earth2studio.run import deterministic


def run_forecast(
    *,
    time,
    nsteps,
    prognostic,
    data,
    io,
):
    """
    Execute a deterministic Earth2Studio forecast.
    """

    if isinstance(time, str):
        time = [datetime.fromisoformat(time)]

    elif isinstance(time, datetime):
        time = [time]

    return deterministic(
        time=time,
        nsteps=nsteps,
        prognostic=prognostic,
        data=data,
        io=io,
    )