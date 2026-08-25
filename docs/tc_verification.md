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
    --output outputs/verification/graphcast_20260811_hernan \
    --plots \
    --plot-output outputs/verification/graphcast_20260811_hernan/plots \
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
