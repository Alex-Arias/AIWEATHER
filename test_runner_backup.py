from aiweather.runners import create_runner
from aiweather.forecast import ForecastRequest

request = ForecastRequest(
    model="graphcast",
    datasource="gfs",
    init_time="2026-07-24T00:00:00",
    lead_time=240,
)

runner = create_runner(request.model)

runner.run(request)

print("\nSUCCESS")