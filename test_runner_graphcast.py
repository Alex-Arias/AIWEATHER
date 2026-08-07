from aiweather.forecast import ForecastRequest
from aiweather.runners import GraphCastRunner

request = ForecastRequest(
    model="graphcast",
    datasource="gfs",
    init_time="2026-07-24T00:00:00",
    lead_time=240,
    output_path="outputs",
    device="cuda",
)

runner = GraphCastRunner()

runner.run(request)

print()
print("SUCCESS")