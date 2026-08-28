# AIWeather

**AIWeather** is a Python framework for AI-based weather forecasting,
tropical cyclone tracking, verification, and model benchmarking.

The project provides a common workflow for running and evaluating AI
weather prediction models while keeping model execution, forecast
storage, tropical cyclone tracking, observational verification, and
scientific diagnostics separated into reusable components.

Current version: **0.1.0**

---

## Current status

AIWeather currently provides an end-to-end tropical cyclone
verification workflow built around standardized AIWeather forecast
stores.

The GraphCast, AIFS2, and Pangu3 workflows have been validated end to end, including:

- standardized forecast storage;
- tropical cyclone tracking;
- multiple tracking methods;
- automatic IBTrACS best-track resolution;
- forecast/best-track temporal alignment;
- track-error verification;
- common-overlap comparison;
- verification CSV exports;
- automated diagnostic plots;
- tropical cyclone field-sequence plots;
- canonical verification case naming;
- run provenance manifests;
- command-line execution.

GraphCast, AIFS2, and Pangu3 are currently evaluated through the same
forecast -> tracking -> QC -> verification framework.

The current controlled Eastern Pacific benchmark includes four
tropical cyclones:

| Storm | Initialization |
|---|---|
| Elida | 2026-07-14 12 UTC |
| Fausto | 2026-07-19 00 UTC |
| Genevieve | 2026-07-24 00 UTC |
| Hernan | 2026-08-11 00 UTC |

The generalized AIFS2 storm-relative wave workflow has been validated
for Elida, Fausto, Genevieve, and Hernan.

Additional AI weather prediction models will be added incrementally
after validation against the existing benchmark workflow.

---

## Documentation

Detailed workflow documentation is available in `docs/`:

- [Quick start](docs/quickstart.md)
- [Running forecast models](docs/forecast_models.md)
- [HPC workflow](docs/hpc_workflow.md)
- [Single-storm TC verification](docs/tc_verification.md)
- [Multi-storm batch verification](docs/batch_verification.md)
- [GraphCast vs AIFS2 comparison](docs/model_comparison.md)
- [AIFS2 wave analysis](docs/wave_analysis.md)
- [Troubleshooting](docs/troubleshooting.md)

---

## Installation

AIWeather currently requires Python 3.11 or newer.

For development, activate the project environment and install the
package in editable mode:

```bash
conda activate aiweather17_pangu3

cd /path/to/AIWeather

python -m pip install -e .
```

Verify the installation:

```bash
aiweather --version
```

Example output:

```text
aiweather 0.1.0
```

The command-line interface can be inspected with:

```bash
aiweather --help
```

---

## Forecast generation

AIWeather provides a unified forecast launcher for the currently supported
deterministic models:

```bash
python scripts/run_aiweather_forecast.py --help
```

Forecasts are written to the standardized layout:

```text
outputs/<model>/<initialization>/forecast.zarr
```

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

See [Running forecast models](docs/forecast_models.md) for details.

---

## Tropical cyclone verification

The primary operational CLI workflow is:

```bash
aiweather verify-tc
```

It performs tropical cyclone tracking and verifies the resulting
forecast tracks against IBTrACS best-track observations.

A complete example is:

```bash
aiweather verify-tc \
    --forecast outputs/graphcast/20260724T000000/forecast.zarr \
    --sid 2026204N08267 \
    --lat-min 5 \
    --lat-max 35 \
    --lon-min -130 \
    --lon-max -90 \
    --device cpu \
    --minimum-overlap 6 \
    --maximum-mean-error-km 450 \
    --plots \
    --field-leads 54 78 96 120
```

In this example:

- the forecast is initialized at `2026-07-24 00:00 UTC`;
- `2026204N08267` is the IBTrACS storm identifier;
- the tracking domain is 5–35°N and 130–90°W;
- tropical cyclone tracking is run on CPU in this example;
- at least six overlapping forecast/best-track points are required;
- tracker candidates with mean reference-track error above 450 km are rejected;
- the standard verification plots are generated;
- field diagnostics are generated at forecast lead times 54, 78, 96,
  and 120 hours.

Run:

```bash
aiweather verify-tc --help
```

for all available options.

---

## Automatic IBTrACS management

An explicit IBTrACS file is not required for the standard workflow.

When `--ibtracs-path` is omitted, AIWeather automatically resolves the
appropriate IBTrACS dataset using its local cache manager.

The behavior can be controlled with:

```text
--ibtracs-path
--ibtracs-basin
--ibtracs-cache-dir
--ibtracs-max-age-hours
--ibtracs-force-update
```

For example, the Eastern Pacific dataset may be cached as:

```text
data/verification/ibtracs/ibtracs.EP.list.v04r01.csv
```

---

## Verification output

When `--output` is omitted, AIWeather automatically constructs a
canonical verification case directory from the storm SID and forecast
initialization time.

For the example above:

```text
results/
└── verification/
    └── 2026204N08267_20260724T000000/
```

A complete verification case contains products such as:

```text
2026204N08267_20260724T000000/
├── common_native.csv
├── common_vitart.csv
├── common_wuduan.csv
├── ibtracs.csv
├── native.csv
├── run_manifest.json
├── verification_native.csv
├── verification_summary.csv
├── verification_vitart.csv
├── verification_wuduan.csv
├── vitart.csv
├── wuduan.csv
└── plots/
    ├── field_sequence.png
    ├── pressure_evolution.png
    ├── track_error.png
    ├── track_error_common.png
    ├── track_map.png
    └── wind_evolution.png
```

### Track products

The tracker-specific files contain the forecast tropical cyclone
tracks generated by the available tracking methods.

Current workflow products include:

```text
native.csv
wuduan.csv
vitart.csv
```

### Verification products

The corresponding verification files contain forecast-versus-IBTrACS
track comparisons:

```text
verification_native.csv
verification_wuduan.csv
verification_vitart.csv
```

`verification_summary.csv` provides case-level verification statistics.

The `common_*.csv` products restrict comparisons to common temporal
overlap, allowing tracking methods to be compared over equivalent
verification periods.

---

## Verification plots

With:

```bash
--plots
```

AIWeather generates the standard tropical cyclone verification
diagnostics.

These currently include:

- forecast and observed track map;
- track error versus lead time;
- common-overlap track error;
- minimum-pressure evolution;
- maximum-wind evolution;
- multi-panel tropical cyclone field sequence.

Field-sequence lead times can be selected with:

```bash
--field-leads 54 78 96 120
```

---

## Run provenance

Each exported verification case includes:

```text
run_manifest.json
```

The manifest records the information needed to identify and reproduce
the verification run, including:

- AIWeather version;
- forecast path;
- forecast identifier;
- model name and model version;
- backend;
- forecast initialization time;
- IBTrACS dataset and basin;
- storm SID;
- tracking domain;
- compute device;
- minimum overlap;
- plot configuration;
- field-sequence lead times;
- manifest creation time.

Example:

```json
{
    "aiweather_version": "0.1.0",
    "forecast": {
        "backend": "earth2studio",
        "forecast_id": "graphcast_gfs_20260724T000000_240h",
        "initialization_time": "2026-07-24T00:00:00",
        "model_name": "graphcast"
    },
    "ibtracs": {
        "basin": "EP"
    },
    "verification": {
        "device": "cuda",
        "sid": "2026204N08267"
    },
    "schema_version": 1
}
```

The manifest is intended to evolve as AIWeather's benchmarking and
experiment-management capabilities expand.

---

## Python API

The verification workflow is also available through the Python API:

```python
from aiweather.verification import (
    run_tc_verification_pipeline,
)

result = run_tc_verification_pipeline(
    "outputs/graphcast/20260724T000000/forecast.zarr",
    sid="2026204N08267",
    lat_min=5.0,
    lat_max=35.0,
    lon_min=-130.0,
    lon_max=-90.0,
    device="cuda",
    minimum_overlap=3,
    generate_plots=True,
    field_lead_times=[
        54,
        78,
        96,
        120,
    ],
)
```

This is the same scientific pipeline used by the command-line
interface.

---

## Testing

Run the standard non-GPU test suite with:

```bash
pytest -q -m "not gpu"
```

GPU-specific tests are marked separately:

```bash
pytest -q -m "gpu"
```

Tests requiring external models, datasets, or system resources use the
`integration` marker.

Before committing changes, the recommended minimum checks are:

```bash
pytest -q -m "not gpu"

git diff --check
git status
```

---

## Development environment

Major dependency upgrades should be tested in a separate experimental
environment before being introduced into the stable AIWeather
development environment.

This is particularly important for dependencies such as Earth2Studio,
CUDA-related libraries, and AI model packages.

The current development environment used for the validated GraphCast,
AIFS2, tropical-cyclone verification, and AIFS2 wave workflows is:

```text
aiweather17_aifs2
```

Environment specifications should be kept synchronized with stable
AIWeather releases so experiments can be reproduced later.

---

## Project structure

The package is organized into components for:

```text
aiweather/
├── backends/
├── data/
├── datasets/
├── forecast/
├── models/
├── output/
├── tracking/
└── verification/
```

The architecture is designed so that model-specific execution remains
separate from common forecast, tracking, and verification
infrastructure.

This separation will allow additional AI weather prediction models to
use the same downstream verification framework.

---

## Roadmap

GraphCast and AIFS2 now provide validated end-to-end implementations
of the common AIWeather forecast, tracking, QC, and verification
workflow.

The current Eastern Pacific benchmark includes Elida, Fausto,
Genevieve, and Hernan. The generalized AIFS2 wave workflow has been
validated for all four storms.

The next development stages are:

1. expand the Eastern Pacific tropical-cyclone sample;
2. introduce and validate the next AI forecast model;
3. apply the same tracking, QC, and verification protocol to that model;
4. expand standardized multi-model and lead-time skill comparisons;
5. add additional meteorological and intensity diagnostics;
6. improve experiment, provenance, and automated reporting tools.

The long-term goal is to provide a reproducible framework in which
multiple AI weather prediction systems can be evaluated through a
common scientific workflow without changing the verification
methodology for each model.

---

## License

AIWeather is released under the MIT License.
