# Running forecast models

AIWeather uses model-specific adapters behind a common forecast interface.
All supported deterministic models write forecasts to the same standardized
storage layout, allowing the same tropical-cyclone verification workflow
to be reused across models.

## Supported deterministic models

The current deterministic forecast workflow supports:

- GraphCast
- AIFS2
- Pangu3
- Pangu6

The preferred forecast entry point is:

```bash
python scripts/run_aiweather_forecast.py --help
```

## Standard output layout

```text
outputs/
├── graphcast/
│   └── <initialization>/forecast.zarr
├── aifs2/
│   └── <initialization>/forecast.zarr
├── pangu3/
│   └── <initialization>/forecast.zarr
└── pangu6/
    └── <initialization>/forecast.zarr
```

The downstream verification layer consumes the standardized
`forecast.zarr` store and does not depend on which model generated it.

## Current benchmark initializations

```text
20260714T120000  Elida
20260719T000000  Fausto
20260724T000000  Genevieve
20260811T000000  Hernan
```

## Example: Genevieve

Initialization:

```text
20260724T000000
```

Forecast horizon:

```text
240 h
```

### GraphCast

```bash
python scripts/run_aiweather_forecast.py \
    graphcast \
    20260724T000000 \
    --datasource gfs \
    --lead-time 240 \
    --device cuda
```

Expected output:

```text
outputs/graphcast/20260724T000000/forecast.zarr
```

### AIFS2

```bash
python scripts/run_aiweather_forecast.py \
    aifs2 \
    20260724T000000 \
    --datasource ifs \
    --datasource-source azure \
    --lead-time 240 \
    --device cuda
```

Expected output:

```text
outputs/aifs2/20260724T000000/forecast.zarr
```

The IFS datasource supports explicit source selection through
`--datasource-source`. The Azure source is shown here because it has been
used successfully for historical AIFS2 runs. If the option is omitted,
the datasource backend default is preserved.

### Pangu3

```bash
python scripts/run_aiweather_forecast.py \
    pangu3 \
    20260724T000000 \
    --datasource gfs \
    --lead-time 240 \
    --device cuda
```

Expected output:

```text
outputs/pangu3/20260724T000000/forecast.zarr
```

Pangu3 uses a 3-hour output cadence.

### Pangu6

```bash
python scripts/run_aiweather_forecast.py \
    pangu6 \
    20260724T000000 \
    --datasource gfs \
    --lead-time 240 \
    --device cuda
```

Expected output:

```text
outputs/pangu6/20260724T000000/forecast.zarr
```

Pangu6 uses a 6-hour output cadence. Earth2Studio implements Pangu3 and Pangu6 hierarchically: they share the same 6-hour and 24-hour forecast trajectory at common valid times, while Pangu3 adds intermediate 3-hour states.

Downstream workflows should always use the timestamps stored in the forecast rather than assuming a fixed cadence.

## Environment

The current combined GraphCast/AIFS2/Pangu3/Pangu6 development environment is:

```bash
conda activate aiweather17_pangu3
```

Model-specific environments can still be retained for reproducibility or
dependency isolation.

## Validate forecast stores

Before tropical-cyclone verification, confirm that the expected stores
exist:

```bash
for model in graphcast aifs2 pangu3 pangu6; do
    test -d outputs/${model}/20260724T000000/forecast.zarr \
        && echo "${model}: OK" \
        || echo "${model}: MISSING"
done
```

The important reproducibility rule is that downstream verification receives
the standardized `forecast.zarr`, regardless of model-specific inference
details.
