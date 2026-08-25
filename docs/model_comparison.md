# GraphCast vs AIFS2 four-storm comparison

The comparison script reads point-level batch verification products. It does not rerun forecasts or cyclone tracking.

## Inputs

```text
results/verification/batch/graphcast_epac_4storm/batch_points.csv
results/verification/batch/aifs2_epac_4storm/batch_points.csv
```

## Run

```bash
python scripts/compare_graphcast_aifs2_tc.py
```

## Output directory

```text
results/verification/comparison/graphcast_vs_aifs2_epac_4storm/
```

Important tables:

```text
paired_track_error.csv
paired_track_error_summary.csv
tracker_sensitivity_summary.csv
```

Important figures include:

```text
graphcast_vs_aifs2_tracks_by_storm.png
graphcast_vs_aifs2_track_error_by_storm.png
graphcast_minus_aifs2_track_error.png
graphcast_vs_aifs2_native_tracks_by_storm.png
graphcast_vs_aifs2_native_track_error_by_storm.png
graphcast_vs_aifs2_all_tracks_by_storm.png
native_vs_wuduan_track_error_by_model_storm.png
```

PDF versions are also generated for the comparison figures.

## Primary comparison

The primary model comparison uses:

```text
WuDuan tracker
full tracker coverage for individual summaries
common paired points for direct GraphCast-AIFS2 differences
```

A paired comparison requires both models to have an accepted track at the same verification time.

Therefore:

```text
full coverage != paired/common coverage
```

For example, a storm can have accepted AIFS2 and GraphCast tracks of different lengths while only a shorter interval is valid for point-by-point model comparison.

## Tracker sensitivity

Native and WuDuan are also compared within each model. This is a scientific diagnostic: large disagreement can indicate loss of storm identity or association with a different pressure minimum.

## Current four-storm interpretation

- Elida supports a paired comparison.
- Fausto supports a paired comparison.
- Genevieve currently has no accepted GraphCast WuDuan match under the reference-based criterion, so it is not included in the paired model statistic.
- Hernan supports a short paired comparison.

Do not convert Genevieve's rejected GraphCast association into an artificial large paired error.
