# Operational Tropical Cyclone Workflow

## 1. Purpose and scope

This document describes the AIWeather workflow for deterministic, operational-style analysis of an already identified tropical cyclone. It covers forecast generation, existing-TC tracking, provenance, deterministic multi-model comparison, AIFS2 storm-relative wave diagnostics, dateline-safe processing, and interpretation of the resulting products.

The workflow is distinct from genesis detection and from formal forecast verification. It is intended to answer the operational question: given a known tropical cyclone and a forecast initialization, how do the supported AI forecast systems represent the subsequent cyclone track, intensity-related fields, and, when available, the storm-relative wave environment?

The workflow has subsequently been extended beyond the original Marie/Lowell implementation to support repeated operational forecast cycles, Native/WuDuan/Vitart tracker comparison, multicycle track visualization, automated AIFS2 operational wave analysis, multicycle wave comparison, and verification of frozen operational tracks against an independent best-track reference.

The current operational architecture therefore separates three complementary products:

1. **forecast diagnostics**, which can be generated immediately from each forecast cycle;
2. **multicycle evolution**, which measures how successive forecasts and tracker solutions change through time; and
3. **formal verification**, which is performed against independent best-track information when sufficient reference observations are available.

## 2. Development history

The operational workflow emerged from earlier tropical-cyclone verification and diagnostic experiments.

### 2.1 Iselle and Douglas development experiments

The Iselle and Douglas experiments were used during workflow development to investigate tracker behavior, vortex organization, continuity, and the relationship between a model-resolved tropical circulation and the time at which a tracking algorithm accepts a persistent center.

These experiments should be treated as developmental and diagnostic work. They helped establish interpretation rules used by the later operational workflow, including the distinction between physical vortex structure and tracker acceptance. Experimental pre-genesis scripts are therefore not part of the production operational interface described in this document.

A tracker failing to accept a center at a given lead does not demonstrate that the model lacks a physically organized circulation. Conversely, a tracked center should not automatically be interpreted as observed genesis or as proof of forecast skill.

### 2.2 Transition to existing-TC tracking

For a storm that is already identified, a genesis-first workflow is unnecessary. AIWeather therefore developed an existing-TC workflow that starts from a known storm position and follows the modeled vortex forward in time while enforcing spatial and dynamical continuity criteria.

This capability was subsequently exercised in the Karina experiment and extended in the Marie and Lowell experiments.

### 2.3 Karina demonstration

Tropical Storm Karina (EP112026), initialized at 1200 UTC 29 August 2026, provided the first completed pseudo-operational demonstration of the newer workflow.

The completed Karina deterministic operational comparison contained AIFS2, GraphCast, and Pangu3 track products. Pangu6 is part of the general AIWeather forecast architecture but was not part of the completed Karina comparison and should not be inserted retrospectively into Karina result tables without generating the corresponding product.

The Karina experiment also established the operational interpretation of AIFS2 wave diagnostics: storm-relative metrics are the primary indicators of cyclone-associated wave conditions, whereas unrestricted regional maxima may be remote from the cyclone.

### 2.4 Marie and Lowell extension

Marie (EP132026) and Lowell (EP122026), initialized at 0000 UTC 3 September 2026, extended the workflow to simultaneous tropical cyclones and four deterministic forecast configurations: AIFS2, GraphCast, Pangu3, and Pangu6.

The experiment also required continuous-longitude handling for Lowell and generalized the AIFS2 wave analysis to operational storm centers.

### 2.5 Odalys/Polo multicycle operational experiment

The Odalys and Polo experiment extended the workflow from isolated operational cases to a sequence of consecutive forecast cycles. Four forecast initializations were retained as separate operational experiments:

- **Cycle 1:** 20 September 2026 at 1200 UTC
- **Cycle 2:** 21 September 2026 at 1200 UTC
- **Cycle 3:** 22 September 2026 at 1200 UTC
- **Cycle 4:** 23 September 2026 at 1200 UTC

The experiment uses the available deterministic forecasts from **GraphCast, AIFS2, Pangu3, and Pangu6** and compares three tropical-cyclone tracking methods: **Native, WuDuan, and Vitart**.

Tracker availability is not identical for every combination of storm, model, cycle, and tracking method. Missing tracker products remain explicitly missing rather than being synthesized or substituted. This preserves the actual operational behavior of each forecast/tracker combination.

Two complementary visualization levels are used:

1. a **single-cycle tracker comparison**, which compares Native, WuDuan, and Vitart solutions for one forecast initialization; and
2. a **multicycle tracker comparison**, which overlays successive forecast cycles while preserving both forecast-model and tracker identity.

In the multicycle representation, forecast-model identity is encoded by color and forecast-cycle identity by line style, while Native, WuDuan, and Vitart solutions are separated by panel. This permits forecast evolution across successive initializations to be examined without treating inter-cycle differences as forecast error.

The multicycle plotter is not restricted to the original four cycles. Additional operational initializations can be appended as they become available, allowing the experiment to continue while a storm remains active.

For AIFS2, the operational wave workflow was also generalized to operate from the tracked cyclone center for each forecast cycle. The cycle-level wrapper discovers available storms, determines compatible forecast and track leads, runs the storm-relative wave diagnostics, and can update the multicycle wave comparison.

These track and wave products can therefore be generated while a cyclone is still evolving. They are **forecast-evolution diagnostics**. Formal statements about forecast skill require comparison with an independent best-track reference over the corresponding verification period.

## 3. Workflow architecture

The operational sequence is:

```text
Storm information / initialization
              |
              v
       Forecast generation
              |
     +--------+--------+--------+
     |        |        |        |
   AIFS2  GraphCast  Pangu3   Pangu6
     |        |        |        |
     +--------+--------+--------+
              |
              v
      Existing-TC tracking
              |
              v
       Track + provenance
              |
              +---------------------------+
              |                           |
              v                           v
   Deterministic multi-model      AIFS2 wave diagnostics
          comparison                      |
              |                           +-- storm-relative SWH
              +-- track map               +-- near-storm wind
              +-- separation              +-- wave evolution
              +-- pressure                +-- radial profiles
              +-- wind                    +-- radius-time Hovmöller
```

The stages should remain conceptually separate. Forecast generation produces model fields; tracking diagnoses the modeled cyclone; comparison quantifies differences among deterministic forecasts; wave analysis diagnoses the AIFS2 wave field relative to the moving modeled center; verification requires an independent reference.

## 4. Forecast models and data sources

The operational framework supports deterministic forecasts from AIFS2, GraphCast, Pangu3, and Pangu6 through the common AIWeather forecast interface.

Typical forecast generation uses:

```bash
python scripts/run_aiweather_forecast.py MODEL INIT_TIME [options]
```

The exact command-line options and datasource should be recorded with each experiment.

Pangu3 and Pangu6 are temporal-sampling configurations within the Pangu workflow. Their differences are useful for cadence-sensitivity analysis, but agreement between them is not equivalent to independent-model consensus.

AIFS2 is currently the operational model used by this workflow for the coupled atmospheric/wave diagnostic products because the required wave fields are available in its forecast output.

## 5. Existing-TC initialization and tracking

### 5.1 Search seed versus model center

The known storm position is used as a spatial search seed for the tracker. It does not force the model center.

After the search is initialized, the accepted center and associated diagnostics are derived from the forecast fields. In particular:

- the modeled cyclone center is model-derived;
- minimum sea-level pressure is model-derived;
- 10-m wind is model-derived;
- subsequent forecast positions are model-derived.

This distinction must be retained in reports and figures. The operational storm position supplies spatial context for finding the initial modeled vortex; it is not a prescribed trajectory.

### 5.2 Track extraction

Existing-TC products are exported with:

```bash
python scripts/export_existing_tc_track.py [options]
```

The workflow records the accepted track and provenance needed to reconstruct the experiment.

Track continuity criteria are intended to prevent implausible jumps between unrelated pressure or wind features. A track ending before the forecast store ends means that no continuation was accepted under the configured tracking criteria. It does not, by itself, establish physical cyclone dissipation.

### 5.3 Provenance

For each case, preserve at minimum:

- storm identifier;
- nominal forecast initialization;
- seed position and its source;
- forecast model and datasource;
- forecast-store path;
- tracking configuration;
- accepted forecast leads;
- software commit;
- output paths.

This information is essential for distinguishing strict real-time, pseudo-operational retrospective, and purely retrospective experiments.

## 6. Operational track summaries and deterministic comparison

Individual track products can be summarized with:

```bash
python scripts/summarize_operational_tracks.py [options]
```

Multi-model deterministic comparisons are generated with:

```bash
python scripts/compare_operational_tc_tracks.py [options]
```

The comparison products include track maps and diagnostics such as pressure evolution, wind evolution, and pairwise track separation at common forecast leads.

### 6.1 Interpretation of model separation

Without an independent reference track, pairwise distance is **inter-model forecast separation**.

It must not be labeled as:

- forecast error;
- track error;
- accuracy;
- skill.

Formal verification requires an independent reference such as finalized best-track data and a common-lead comparison procedure.

A model that continues longer than another model under the tracker should likewise not be described as more accurate solely because it has greater tracker coverage.

### 6.2 Forecast-cycle comparison

Forecast cycles for the same tropical cyclone can be compared with
`scripts/compare_operational_tc_cycles.py`.

The comparator aligns tracks by exact absolute valid time rather than
equal forecast lead. Same-model cycle displacement therefore measures
the change in the forecast trajectory between initialization cycles; it
is a forecast revision diagnostic, not a track error.

The diagnostic writes cycle-displacement and inter-system-spread CSV
products together with track, displacement, and matched-spread figures.
For quantitative comparison of spread between cycles, the
`matched_system_spread` products should be used because they restrict
both cycles to exactly the same valid-time window.

The `--spread-models` option explicitly selects the systems included in
inter-system spread. This permits scientifically related configurations
to remain in the track and cycle-displacement diagnostics without
counting them as independent consensus members.

Inter-system spread measures deterministic forecast diversity, not
forecast error or skill. Statements about accuracy or skill require an
independent reference.

### 6.3 Operational tracker comparison

AIWeather provides two complementary plotting tools for inspecting
operational tropical-cyclone tracks.

#### Single-cycle comparison

For one forecast initialization, the Native, WuDuan, and Vitart
tracker solutions can be compared across storms and forecast models.

```bash
python scripts/plot_operational_tracker_comparison.py \
    --init 20260923T120000 \
    --storms Odalys Polo \
    --output results/operational/cycle4_tracker_comparison.png
```

The figure uses rows for storms, columns for Native, WuDuan, and
Vitart trackers, and colors for GraphCast, AIFS2, Pangu3, and Pangu6.
Missing tracker products remain explicitly missing.

The first four Odalys/Polo cycles are:

- Cycle 1: `20260920T120000`
- Cycle 2: `20260921T120000`
- Cycle 3: `20260922T120000`
- Cycle 4: `20260923T120000`

#### Multicycle comparison

Successive operational forecast cycles can be overlaid with:

```bash
python scripts/plot_multicycle_tracker_comparison.py \
    --inits 20260920T120000 20260921T120000 \
            20260922T120000 20260923T120000 \
    --storms Odalys Polo \
    --output results/operational/multicycle_tracker_comparison.png
```

Initializations should be supplied from earliest to latest.

The multicycle figure preserves three independent dimensions of the
experiment:

- color = forecast model;
- line style = forecast cycle; and
- panel = tracking method.

For the original experiment, Cycles 1 through 4 correspond to
20–23 September 2026 at 1200 UTC. Additional operational cycles can
be appended as they become available without changing the analysis
structure.

These figures are forecast-diagnostic products. Differences among
models, trackers, or successive forecast cycles describe forecast
evolution and tracker sensitivity; they are not forecast-error
metrics. Formal forecast error and skill assessment require
comparison against an independent verifying best track.

## 7. AIFS2 storm-relative wave analysis

The operational AIFS2 wave workflow diagnoses the wave field relative
to the moving tracked cyclone center. Wave diagnostics are forecast
products and can therefore be generated while a tropical cyclone is
still active; they do not require a completed best-track record.

### 7.0 Operational cycle wrapper

The preferred entry point for a complete operational forecast cycle is:

```bash
python scripts/run_operational_waves.py \
    --init YYYYMMDDTHHMMSS
```

The wrapper discovers the storms available for the requested
initialization and runs the AIFS2 storm-relative wave workflow for
each case.

Before generating products, the same workflow can be checked with:

```bash
python scripts/run_operational_waves.py \
    --init YYYYMMDDTHHMMSS \
    --dry-run
```

Dry-run mode discovers the operational storms, identifies the AIFS2
forecast and track files, determines the compatible forecast and
track leads, and runs analyzer preflight without generating wave
products.

The operational wave leads are therefore determined from the data
available for each storm rather than assumed to be identical among
cases.

After successful storm analyses, the wrapper updates the multicycle
wave comparison. This final step can be disabled when required with:

```bash
python scripts/run_operational_waves.py \
    --init YYYYMMDDTHHMMSS \
    --skip-comparison
```

The underlying analysis components remain available individually.
The principal scripts are:

- `scripts/analyze_aifs2_waves.py` for storm-relative wave analysis;
- `scripts/plot_aifs2_wave_evolution.py` for wave-evolution figures;
- `scripts/plot_aifs2_wave_hovmoller.py` for radius-time diagnostics;
- `scripts/plot_multicycle_wave_comparison.py` for comparison among
  operational forecast cycles;
- `scripts/wave_cases.py` for operational wave-case definitions; and
- `scripts/wave_centers.py` for storm-center handling.

The operational sequence is therefore:

```text
AIFS2 forecast + operational tracked cyclone center
                         |
                         v
              operational preflight
                         |
                         v
       compatible forecast/track leads
                         |
                         v
          storm-relative wave analysis
                         |
              +----------+----------+
              |                     |
              v                     v
       per-cycle diagnostics   wave evolution /
                               radial structure
              |
              v
       multicycle comparison
```

### 7.1 Storm-relative versus regional extrema

The maximum significant wave height within a specified radius of the tracked center, commonly 300 km in the completed operational experiments, is the principal cyclone-associated SWH metric.

An unrestricted regional maximum can occur far from the cyclone because of another weather system, remote swell, or environmental wave structure. Regional maxima must therefore be accompanied by their distance from the tracked center before they are attributed to the cyclone.

### 7.2 Radius-time Hovmöller diagnostics

The radius-time Hovmöller organizes wave statistics by distance from the moving cyclone center and forecast lead. It is useful for identifying:

- the radial location of enhanced wave conditions;
- temporal changes in the storm-relative wave field;
- outward propagation or broadening of wave energy;
- differences between instantaneous local maxima and azimuthally averaged structure.

The Hovmöller complements, rather than replaces, storm-relative maps and 300-km extrema.

## 8. Dateline-safe processing

Cyclones approaching the 180-degree meridian require a continuous longitude representation.

AIWeather maps longitudes to a branch centered near the storm reference longitude so that the track and wave fields do not acquire an artificial plotting discontinuity at the dateline.

For example:

```text
-190 degrees == 170 degrees E
```

Longitudes outside the conventional `[-180, 180]` range may therefore appear intentionally in operational plots.

Lowell provided the principal operational test of this behavior. Its track and wave products were plotted on a continuous longitude branch, preventing a false line across the map.

## 9. Operational case studies

### 9.1 Karina

Karina was initialized at 1200 UTC 29 August 2026.

The completed operational comparison included AIFS2, GraphCast, and Pangu3. The experiment demonstrated existing-TC tracking, deterministic inter-model comparison, and AIFS2 storm-relative wave analysis.

The maximum AIFS2 SWH within 300 km of the tracked center was approximately 7.907 m at +126 h. A larger regional maximum occurred far from Karina and was therefore not automatically attributed to the cyclone.

Because the exact 12 UTC analyzed position used as the search seed was published in a later advisory, the experiment is described as **pseudo-operational retrospective**, not strict real-time.

### 9.2 Marie

Marie was initialized at 0000 UTC 3 September 2026.

The completed deterministic products included AIFS2, GraphCast, Pangu3, and Pangu6. Accepted tracks extended to +210 h, +180 h, +225 h, and +222 h, respectively, under the configured tracking criteria.

For AIFS2, maximum SWH within 300 km reached approximately 9.612 m at +48 h. Maximum 10-m wind within 300 km was approximately 26.368 m s-1 at +48 h. The radius-time analysis reached a maximum radial-bin mean SWH of approximately 7.98 m near 75 km radius at +48 h.

#### Marie forecast-cycle experiment

A second Marie experiment was initialized at 0000 UTC 5 September 2026
using 20.6 degrees N, 118.7 degrees W as the NHC spatial search seed.
Because that analyzed position was issued at 0300 UTC, the experiment is
classified as pseudo-operational retrospective.

The 3 September cycle was preserved unchanged as the baseline, while the
5 September cycle was treated as an independent later-cycle experiment
for AIFS2, GraphCast, Pangu3, and Pangu6.

Cycle-to-cycle tracks were compared at exact common absolute valid times.
All four forecast configurations shifted toward substantially more
poleward later-cycle trajectories as the forecasts evolved. The largest
cycle displacements occurred in the Pangu configurations. These
differences represent forecast-cycle revision, not forecast error.

For the inter-system spread calculation, AIFS2, GraphCast, and Pangu3
were selected explicitly with `--spread-models aifs2 graphcast pangu3`.
Pangu6 remained in the track-map and cycle-displacement diagnostics but
was not counted as an additional independent system because Pangu3 and
Pangu6 are temporal-sampling configurations of the same Pangu workflow.

Across the 23 exact valid times shared by both cycles and the three
selected systems, from 0000 UTC 5 September through 1200 UTC
10 September, mean pairwise track spread decreased from approximately
301.4 km in the 3 September cycle to 151.0 km in the 5 September cycle,
a reduction of approximately 49.9 percent. Maximum pairwise separation
over the same matched window decreased from approximately 1032.1 km to
391.2 km.

The later cycle therefore showed substantially greater deterministic
inter-system agreement over the matched valid-time window. This does
not by itself demonstrate improved forecast skill; independent
best-track verification is required to assess accuracy.

### 9.3 Lowell

Lowell was initialized at 0000 UTC 3 September 2026.

Accepted tracks extended to +240 h for AIFS2, +204 h for GraphCast, and +222 h for both Pangu3 and Pangu6 under the configured tracking criteria.

For AIFS2, maximum SWH within 300 km reached approximately 9.720 m at +132 h. Maximum 10-m wind within 300 km was approximately 31.664 m s-1 at +0 h. The maximum radial-bin mean SWH was approximately 8.70 m near 25 km radius at +30 h.

Lowell also validated the continuous-longitude plotting and analysis path for a dateline-adjacent forecast.

The exact 00 UTC seed positions for Marie and Lowell were available from advisories issued after the nominal initialization. These cases are therefore also described as pseudo-operational retrospective.

## 10. Scientific interpretation and limitations

The operational workflow is designed to preserve several distinctions that are important for scientific interpretation.

### 10.1 Tracker behavior is not identical to physical storm evolution

Tracker acceptance is an algorithmic diagnosis. Loss of an accepted track does not prove that the modeled circulation has disappeared. Earlier Iselle and Douglas development experiments were important in establishing this distinction.

### 10.2 Model comparison is not verification

AIFS2, GraphCast, Pangu3, and Pangu6 can disagree substantially in track, pressure, or wind evolution. These differences characterize deterministic forecast diversity. They become errors only after comparison with an independent reference.

### 10.3 Pangu cadence experiments are not independent-model evidence

Pangu3 and Pangu6 provide useful information about temporal sampling and workflow sensitivity. They should not be counted as two independent dynamical models when interpreting consensus.

### 10.4 Wave attribution requires storm-relative context

A large SWH anywhere in the forecast domain is not necessarily generated by the tracked tropical cyclone. Distance from the storm center and the temporal/radial evolution of the wave field are required for attribution.

### 10.5 Pseudo-operational versus strict real-time experiments

The operational scripts can support an operational-style workflow, but a case should be called strict real-time only when every input used at initialization was actually available at that time. When later-issued storm-center information is used retrospectively, the case should be labeled pseudo-operational retrospective.

## 11. Recommended workflow for a new storm

A new operational experiment should follow this sequence:

1. Select and freeze the nominal forecast initialization.
2. Record the storm identifier and operational center source.
3. Record whether the center information was available at the nominal initialization.
4. Generate each deterministic forecast in a separate forecast store.
5. Run existing-TC tracking independently for each model.
6. Save track and provenance products.
7. Inspect tracker continuity before interpreting track termination.
8. Generate individual track summaries.
9. Generate the deterministic multi-model comparison.
10. For AIFS2, generate storm-relative wave diagnostics.
11. Generate wave-evolution and radius-time Hovmöller products.
12. Inspect the distance of regional extrema from the cyclone.
13. Apply continuous-longitude handling when the storm approaches the dateline.
14. Add independent best-track and wave observations when formal verification is required.
15. Record the AIWeather commit and environment used for the experiment.

## 12. Reproducibility

The Marie/Lowell operational wave workflow was validated at:

```text
AIWeather commit: 78b23bb
Commit description: Support Marie and Lowell operational wave analysis
Regression status: 466 passed, 88 warnings
```
The subsequent forecast-cycle comparison implementation was validated
with the full GPU regression suite before commit:

Regression status: 472 passed, 88 warnings

Operational products should retain enough provenance to associate a result with its exact forecast store, tracking configuration, case definition, and source-code checkpoint.

The repository source, rather than copied working-tree scripts, should be treated as the authoritative implementation.

## 13. Relationship to other documentation

This document provides the operational orchestration and interpretation layer. Detailed information remains in the specialized documentation:

- `docs/forecast_models.md` — supported forecast-model configurations;
- `docs/tc_verification.md` — tropical-cyclone verification methodology;
- `docs/model_comparison.md` — model-comparison procedures;
- `docs/wave_analysis.md` — AIFS2 wave diagnostics;
- `docs/hpc_workflow.md` — HPC execution practices;
- `docs/troubleshooting.md` — known workflow and interpretation issues;
- `docs/quickstart.md` — general AIWeather entry points.

The operational workflow should be read together with those documents rather than as a replacement for them.
