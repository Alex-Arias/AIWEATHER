# AIWeather Development Notes

## Cluster

- Rocky Linux 9
- SLURM
- GPU partition: oc_gpu
- NVIDIA H200 NVL
- Conda env: earth2studio

## Installed

- Earth2Studio 0.16
- JAX
- CUDA

## Operational workflow

Current objective:
Run GraphCastOperational using deterministic() with GFS.

## AIWeather architecture

ForecastRequest
Runner
Adapter
Forecast
Postprocessing