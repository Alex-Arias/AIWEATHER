# Running forecast models

AIWeather uses model-specific runners but writes forecasts into a common output organization so that the same verification layer can be applied to different AI weather models.

## Standard output layout

```text
outputs/
├── aifs2/
│   └── <initialization>/forecast.zarr
└── graphcast/
    └── <initialization>/forecast.zarr
```

Current four-storm initializations:

```text
20260714T120000  Elida
20260719T000000  Fausto
20260724T000000  Genevieve
20260811T000000  Hernan
```

## AIFS2

The repository contains the AIFS2 forecast script:

```bash
python scripts/run_aifs2.py --help
```

Inspect the current command-line interface before a production run:

```bash
sed -n '1,260p' scripts/run_aifs2.py
```

The validated AIFS2 environment is:

```bash
conda activate aiweather17_aifs2
```

Before running a new storm, confirm that the desired initialization is not already present:

```bash
find outputs/aifs2 \
    -mindepth 1 \
    -maxdepth 1 \
    -type d \
    -printf '%f\n' | sort
```

## GraphCast

GraphCast uses the same AIWeather concept:

```text
initialization + forecast length
             ↓
       model runner
             ↓
 standardized forecast.zarr
             ↓
 common TC verification
```

Use the repository's current GraphCast runner/CLI configuration for production rather than copying an old command from notes. The standardized forecast path should be:

```text
outputs/graphcast/<initialization>/forecast.zarr
```

## Forecast validation

Before TC verification, confirm that the forecast store exists:

```bash
test -d outputs/aifs2/20260811T000000/forecast.zarr && echo OK
test -d outputs/graphcast/20260811T000000/forecast.zarr && echo OK
```

The important reproducibility rule is that downstream verification receives the standardized `forecast.zarr`, regardless of the model-specific forecast-generation details.
