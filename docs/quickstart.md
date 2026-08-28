# Quick start

This page gives the shortest end-to-end AIWeather workflow.

## 1. Enter the repository and activate the environment

```bash
cd /LUSTRE/ariasv/AIWeather
conda activate aiweather17_pangu3
```

## 2. Check the installation

```bash
aiweather --version
aiweather --help
```

For editable development installs:

```bash
python -m pip install -e .
```

## 3. Generate forecasts

Use the unified forecast launcher described in
[forecast_models.md](forecast_models.md).

Example: Genevieve initialized at `2026-07-24 00 UTC`.

GraphCast:

```bash
python scripts/run_aiweather_forecast.py \
    graphcast \
    20260724T000000 \
    --datasource gfs \
    --lead-time 240 \
    --device cuda
```

AIFS2:

```bash
python scripts/run_aiweather_forecast.py \
    aifs2 \
    20260724T000000 \
    --datasource ifs \
    --datasource-source azure \
    --lead-time 240 \
    --device cuda
```

Pangu3:

```bash
python scripts/run_aiweather_forecast.py \
    pangu3 \
    20260724T000000 \
    --datasource gfs \
    --lead-time 240 \
    --device cuda
```

Forecast products are standardized as:

```text
outputs/<model>/<YYYYMMDDTHHMMSS>/forecast.zarr
```

For this example:

```text
outputs/graphcast/20260724T000000/forecast.zarr
outputs/aifs2/20260724T000000/forecast.zarr
outputs/pangu3/20260724T000000/forecast.zarr
```


## 4. Verify one tropical cyclone

Example: GraphCast / Hernan.

```bash
aiweather verify-tc \
    --forecast outputs/graphcast/20260811T000000/forecast.zarr \
    --sid 2026223N14233 \
    --lat-min 5 \
    --lat-max 35 \
    --lon-min -150 \
    --lon-max -100 \
    --device cpu \
    --minimum-overlap 6 \
    --maximum-mean-error-km 450 \
    --output results/verification/graphcast_20260811_hernan \
    --plots \
    --plot-output results/verification/graphcast_20260811_hernan/plots \
    --field-leads 24 48 72 96
```

## 5. Run the four-storm EPAC batch

```bash
python scripts/run_epac_tc_batch.py aifs2 --device cpu
python scripts/run_epac_tc_batch.py graphcast --device cpu
```

Expected batch products include:

```text
batch_summary.csv
batch_aggregate.csv
batch_lead_time.csv
batch_points.csv
```

## 6. Compare GraphCast and AIFS2

```bash
python scripts/compare_graphcast_aifs2_tc.py
```

The comparison reads the batch point tables and does **not** rerun forecasts or tracking.

## 7. Run AIFS2 wave diagnostics

```bash
python scripts/analyze_aifs2_waves.py --storm elida
python scripts/plot_aifs2_wave_evolution.py --storm elida
python scripts/plot_aifs2_wave_hovmoller.py --storm elida
```

Repeat for configured storms and then run:

```bash
python scripts/compare_aifs2_wave_storms.py
```

See [wave_analysis.md](wave_analysis.md) before adding a new storm.
