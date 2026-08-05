"""
Earth2Studio inference backend.
"""

from __future__ import annotations

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

    This wrapper isolates AIWeather from direct Earth2Studio API calls.
    Future versions may support ensemble and stochastic inference
    without changing the runner interface.
    """

    return deterministic(
        time=time,
        nsteps=nsteps,
        prognostic=prognostic,
        data=data,
        io=io,
    )