from aiweather.forecast import ForecastRequest


request = ForecastRequest(
    model="graphcast",
    datasource="gfs",
    init_time="2025-07-01T00:00",
    lead_time=24,
)

print(request)