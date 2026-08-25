# Multi-storm batch verification

The batch layer reuses the tested single-case verification pipeline rather than duplicating tracking or verification logic.

## Current EPAC script

```bash
python scripts/run_epac_tc_batch.py --help
```

Run AIFS2:

```bash
python scripts/run_epac_tc_batch.py aifs2 --device cpu
```

Run GraphCast:

```bash
python scripts/run_epac_tc_batch.py graphcast --device cpu
```

Optional per-case plots:

```bash
python scripts/run_epac_tc_batch.py aifs2 --device cpu --plots
```

## Current controlled configuration

```text
MINIMUM_OVERLAP = 6
MAXIMUM_MEAN_ERROR_KM = 450.0
```

The script contains the four EPAC cases:

```text
Elida
Fausto
Genevieve
Hernan
```

## Batch outputs

The batch result exports:

### `batch_summary.csv`

Per-case/per-tracker verification summary, including tracker QC.

### `batch_aggregate.csv`

QC-filtered aggregate verification statistics.

### `batch_lead_time.csv`

QC-filtered statistics grouped by forecast lead time.

### `batch_points.csv`

Point-level forecast-versus-observation records for each case, tracker, and coverage set.

The point table is the principal input to the GraphCast-vs-AIFS2 plotting/comparison script.

## QC rule

Aggregate statistics should be interpreted only after tracker QC. The benchmark intentionally distinguishes:

```text
forecast skill
tracker availability
tracker/reference association
coverage
```

These are related but not interchangeable quantities.
