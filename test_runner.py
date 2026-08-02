from aiweather.forecast import ForecastRequest
from aiweather.runners import GraphCastRunner

request = ForecastRequest(
    model="graphcast",
    datasource="gfs",
    init_time="2023-10-23T00:00:00",
    lead_time=72,
    output_path="outputs",
    device="cuda",
)

runner = GraphCastRunner()

runner.run(request)

print()
print("SUCCESS")