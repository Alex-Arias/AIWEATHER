import pytest

from aiweather.forecast import ForecastRequest
from aiweather.runners import GraphCastRunner


@pytest.mark.gpu
@pytest.mark.integration
def test_graphcast_gpu():
    """Run a real GraphCast forecast on a GPU node."""

    request = ForecastRequest(
        model="graphcast",
        datasource="gfs",
        init_time="2026-07-24T00:00:00",
        lead_time=240,
        output_path="outputs/test_graphcast.zarr",
        device="cuda",
    )

    runner = GraphCastRunner()
    runner.run(request)
