# AIFS2 tropical-cyclone wave analysis

## Status

The AIFS2 wave workflow is already generalized across cyclone cases. The reusable scripts are:

```text
scripts/wave_cases.py
scripts/analyze_aifs2_waves.py
scripts/plot_aifs2_wave_evolution.py
scripts/plot_aifs2_wave_hovmoller.py
scripts/compare_aifs2_wave_storms.py
```

Storm-specific Fausto and Genevieve copies were intentionally removed after the generalized workflow was validated. New storms should be added through the shared case configuration rather than by cloning analysis scripts.

The existing generalized workflow has been exercised for Elida, Fausto, and Genevieve. Hernan is the next useful validation case.

## Conceptual workflow

```text
AIFS2 forecast.zarr
       +
verified tropical-cyclone track
       ↓
storm/case configuration
       ↓
wave-field extraction
       ↓
forecast/track lead-time alignment
       ↓
storm-relative distance and diagnostics
       ↓
single-storm diagnostic tables
       ↓
wave-evolution plots
       ↓
storm-relative Hovmöller plots
       ↓
multi-storm comparison
```

## 1. Environment

```bash
cd /LUSTRE/ariasv/AIWeather
conda activate aiweather17_aifs2
```

## 2. Check the generalized scripts

```bash
python -m py_compile \
    scripts/wave_cases.py \
    scripts/analyze_aifs2_waves.py \
    scripts/plot_aifs2_wave_evolution.py \
    scripts/plot_aifs2_wave_hovmoller.py \
    scripts/compare_aifs2_wave_storms.py
```

## 3. Inspect available cases

`wave_cases.py` is the central case configuration. A new cyclone should be represented there so that all generalized scripts consume the same storm metadata.

Before adding a case, confirm the forecast store:

```bash
find outputs/aifs2 \
    -mindepth 2 \
    -maxdepth 2 \
    -type d \
    -name forecast.zarr | sort
```

## 4. Generate single-storm diagnostics

Examples:

```bash
python scripts/analyze_aifs2_waves.py --storm elida
python scripts/analyze_aifs2_waves.py --storm fausto
python scripts/analyze_aifs2_waves.py --storm genevieve
```

The analysis writes storm-specific products below:

```text
results/waves/aifs2_<storm>/
```

Existing comparison inputs include:

```text
aifs2_<storm>_wave_diagnostics.csv
aifs2_<storm>_wave_radial_profiles.csv
```

These CSV files should be treated as the stable interface between the numerical analysis and the comparison/plotting layer.

## 5. Plot wave evolution

```bash
python scripts/plot_aifs2_wave_evolution.py --storm elida
python scripts/plot_aifs2_wave_evolution.py --storm fausto
python scripts/plot_aifs2_wave_evolution.py --storm genevieve
```

For all configured three-storm cases:

```bash
for storm in elida fausto genevieve; do
    python scripts/plot_aifs2_wave_evolution.py \
        --storm "$storm"
done
```

The evolution products summarize storm-centered wave behavior through forecast time.

## 6. Plot storm-relative Hovmöller diagnostics

```bash
python scripts/plot_aifs2_wave_hovmoller.py --storm elida
python scripts/plot_aifs2_wave_hovmoller.py --storm fausto
python scripts/plot_aifs2_wave_hovmoller.py --storm genevieve
```

Batch form:

```bash
for storm in elida fausto genevieve; do
    python scripts/plot_aifs2_wave_hovmoller.py \
        --storm "$storm"
done
```

The Hovmöller workflow is storm-relative and uses radial/lead-time organization rather than a fixed geographic point.

## 7. Compare storms

After all required single-storm diagnostic CSVs exist:

```bash
python scripts/compare_aifs2_wave_storms.py
```

The current comparison script reads the Elida, Fausto, and Genevieve diagnostics and radial profiles and writes products under:

```text
results/waves/aifs2_storm_comparison/
```

The existing three-storm products include:

```text
aifs2_epac_3storm_wave_summary.csv
aifs2_epac_3storm_wave_comparison.png
aifs2_epac_3storm_wave_comparison.pdf
```

## 8. Adding Hernan

Hernan should be used to test the documentation rather than to redesign the workflow.

Recommended sequence:

```text
1. Confirm outputs/aifs2/20260811T000000/forecast.zarr.
2. Confirm the accepted Hernan TC track product used by the wave workflow.
3. Add Hernan to scripts/wave_cases.py.
4. Compile all generalized wave scripts.
5. Run analyze_aifs2_waves.py --storm hernan.
6. Inspect the generated diagnostics and radial profiles.
7. Run wave-evolution plotting.
8. Run Hovmöller plotting.
9. Add Hernan to compare_aifs2_wave_storms.py.
10. Rename/update the comparison products from 3storm to 4storm.
11. Rerun the multi-storm comparison.
12. Review figures before committing.
```

Expected commands after the case configuration is added:

```bash
python scripts/analyze_aifs2_waves.py \
    --storm hernan

python scripts/plot_aifs2_wave_evolution.py \
    --storm hernan

python scripts/plot_aifs2_wave_hovmoller.py \
    --storm hernan
```

Then:

```bash
python scripts/compare_aifs2_wave_storms.py
```

## 9. Quality control

Before interpreting a wave case, verify:

- the correct AIFS2 initialization is being read;
- the selected TC track corresponds to the intended storm;
- track and wave forecast times overlap;
- longitude conventions are handled consistently;
- radial distance is storm-relative;
- requested radial and lead-time limits are appropriate for the available track;
- diagnostic CSVs contain the expected lead times before plotting;
- comparison scripts do not silently mix old three-storm and new four-storm products.

## 10. Reproducibility principle

Do **not** create:

```text
analyze_aifs2_waves_hernan.py
plot_aifs2_wave_evolution_hernan.py
plot_aifs2_wave_hovmoller_hernan.py
```

Instead, extend `wave_cases.py` and reuse the generalized scripts. If Hernan requires analysis-code changes rather than case configuration, first determine whether the issue is genuinely physical/data-specific or whether the generalized workflow has an undocumented assumption.

That test is one of the main reasons to process Hernan next.
