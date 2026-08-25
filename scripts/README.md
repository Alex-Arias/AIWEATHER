# Workflow command examples

This file is a compact command sheet. The detailed rationale is in `docs/`.

## TC verification: Hernan / GraphCast

```bash
aiweather verify-tc \
    --forecast outputs/graphcast/20260811T000000/forecast.zarr \
    --sid 2026223N14233 \
    --lat-min 5 --lat-max 35 \
    --lon-min -150 --lon-max -100 \
    --device cpu \
    --minimum-overlap 6 \
    --maximum-mean-error-km 450 \
    --output outputs/verification/graphcast_20260811_hernan \
    --plots \
    --plot-output outputs/verification/graphcast_20260811_hernan/plots \
    --field-leads 24 48 72 96
```

## Four-storm batches

```bash
python scripts/run_epac_tc_batch.py aifs2 --device cpu
python scripts/run_epac_tc_batch.py graphcast --device cpu
```

## Four-storm TC comparison

```bash
python scripts/compare_graphcast_aifs2_tc.py
```

## Generalized AIFS2 wave workflow

```bash
python scripts/analyze_aifs2_waves.py --storm elida
python scripts/plot_aifs2_wave_evolution.py --storm elida
python scripts/plot_aifs2_wave_hovmoller.py --storm elida
python scripts/compare_aifs2_wave_storms.py
```
