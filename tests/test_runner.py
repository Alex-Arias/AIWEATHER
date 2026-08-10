from aiweather.forecast import ForecastRequest
from aiweather.runners import create_runner


def test_runner_factory():
    """Verify that the runner factory can create a GraphCast runner."""

    request = ForecastRequest(
        model="graphcast",
        datasource="gfs",
        init_time="2026-07-24T00:00:00",
        lead_time=240,
        device="cpu",
    )

    runner = create_runner(request.model)

    assert runner is not None
    assert runner.MODEL_NAME == "graphcast"
