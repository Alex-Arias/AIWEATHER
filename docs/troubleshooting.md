# Troubleshooting

## No tropical cyclone genesis was detected

Message:

```text
RuntimeError: No tropical cyclone genesis was detected in the selected forecast region.
```

Current AIWeather behavior permits external tracker evaluation when an external reference track is available even if Native genesis is not detected. In that case the Native track is unavailable rather than fatal to the entire verification.

If no external reference is available, the no-genesis condition remains an error.

## CuPy distance computation test failed

Example:

```text
CuPy distance computation test failed with error:
cuVS >= 24.12 or pylibraft < 24.12 should be installed ...
```

This has appeared during CPU verification runs while the verification itself completed successfully. Treat it as a dependency/backend warning unless the actual tracker or verification fails.

## Earth2Studio `min_size` warning

Example:

```text
FutureWarning: Parameter `min_size` is deprecated ...
```

This warning originates in the Earth2Studio tracking dependency and has been non-fatal in the current workflow.

## Missing forecast store

Check:

```bash
test -d outputs/aifs2/20260811T000000/forecast.zarr && echo OK
```

or:

```bash
find outputs/aifs2 \
    -mindepth 2 \
    -maxdepth 2 \
    -type d \
    -name forecast.zarr | sort
```

## Candidate rejected by reference matching

A rejected candidate is not automatically a forecast with an enormous track error. With:

```text
maximum_mean_error_km = 450
```

a candidate that does not sufficiently resemble the observed storm is excluded from the paired model comparison.

Genevieve/GraphCast is the current four-storm example of this distinction.

## Insufficient paired coverage

Full tracker coverage and paired model coverage are different. Direct GraphCast-AIFS2 differences require common valid times. Always inspect the paired point count before interpreting percentages or mean differences.

## Wave comparison still says `3storm`

The current generalized single-storm wave scripts support reusable cases, but the existing comparison script/products were created for Elida, Fausto, and Genevieve. After Hernan is validated, update the comparison case list and output names consistently to `4storm`.

## Syntax checks before committing

```bash
python -m py_compile \
    scripts/run_epac_tc_batch.py \
    scripts/compare_graphcast_aifs2_tc.py

python -m py_compile \
    scripts/wave_cases.py \
    scripts/analyze_aifs2_waves.py \
    scripts/plot_aifs2_wave_evolution.py \
    scripts/plot_aifs2_wave_hovmoller.py \
    scripts/compare_aifs2_wave_storms.py

git diff --check
```

Then run the relevant tests and inspect generated figures before staging.
