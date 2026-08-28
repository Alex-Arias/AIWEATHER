# Single-storm tropical-cyclone verification

AIWeather verifies a forecast cyclone against an external best-track reference and evaluates multiple tracking approaches.

## Example: Hernan

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

## Important controls

### `--minimum-overlap 6`

A tracker candidate must have sufficient common verification points with the reference trajectory before it can be treated as a useful match.

### `--maximum-mean-error-km 450`

Candidate association is rejected when the mean reference-track error exceeds the configured threshold. This prevents a dynamically plausible but unrelated vortex from being scored as the target storm.

The value used in the current four-storm experiment is:

```text
450 km
```

Keep this fixed when extending the controlled experiment unless the scientific protocol is deliberately changed.

## Native-genesis behavior

AIWeather now supports the following verification path:

```text
Native genesis detected
    ├── yes → Native track + external trackers
    └── no
         ↓
 external reference available?
         ├── yes → Native unavailable; WuDuan/Vitart can still be evaluated
         └── no  → verification cannot use the fallback
```

This distinction is important. A missing Native track is reported as unavailable rather than forcing a false Native result.

## Standard plots

A successful `--plots` run can produce:

```text
track_map.png
track_error.png
track_error_common.png
pressure_evolution.png
wind_evolution.png
field_sequence.png
```

## Interpretation

Always inspect:

1. tracker availability;
2. overlap count;
3. reference matching/QC;
4. forecast horizon represented by the accepted track;
5. full versus common/paired coverage.

Do not interpret a tracker failure as automatically equivalent to a model forecast failure.

## Genevieve three-model example

The same verification workflow can be applied to standardized forecast
stores from GraphCast, AIFS2, and Pangu3.

Genevieve:

```text
SID             2026204N08267
Initialization  20260724T000000
Domain          5–35 N, 130–90 W
```

The following focused example uses:

```text
minimum overlap          3
maximum mean track error 250 km
field leads              54 78 96 120 h
```

These settings are separate from the controlled four-storm benchmark
configuration documented above.

### GraphCast

```bash
aiweather verify-tc \
    --forecast outputs/graphcast/20260724T000000/forecast.zarr \
    --sid 2026204N08267 \
    --lat-min 5 \
    --lat-max 35 \
    --lon-min -130 \
    --lon-max -90 \
    --device cpu \
    --minimum-overlap 3 \
    --maximum-mean-error-km 250 \
    --output results/verification/graphcast_genevieve_20260724T000000 \
    --plots \
    --field-leads 54 78 96 120
```

### AIFS2

```bash
aiweather verify-tc \
    --forecast outputs/aifs2/20260724T000000/forecast.zarr \
    --sid 2026204N08267 \
    --lat-min 5 \
    --lat-max 35 \
    --lon-min -130 \
    --lon-max -90 \
    --device cpu \
    --minimum-overlap 3 \
    --maximum-mean-error-km 250 \
    --output results/verification/aifs2_genevieve_20260724T000000 \
    --plots \
    --field-leads 54 78 96 120
```

### Pangu3

```bash
aiweather verify-tc \
    --forecast outputs/pangu3/20260724T000000/forecast.zarr \
    --sid 2026204N08267 \
    --lat-min 5 \
    --lat-max 35 \
    --lon-min -130 \
    --lon-max -90 \
    --device cpu \
    --minimum-overlap 3 \
    --maximum-mean-error-km 250 \
    --output results/verification/pangu3_genevieve_20260724T000000 \
    --plots \
    --field-leads 54 78 96 120
```

Keep tracker-association criteria fixed across models when making a
controlled comparison unless the scientific protocol is deliberately
changed.

A failed or rejected tracker association is itself a valid verification
outcome. Do not relax the association threshold merely to force a forecast
track. Field-based structural diagnostics can be evaluated separately when
the scientific question requires them.
