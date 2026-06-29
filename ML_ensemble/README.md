# ML Ensemble

Retained ensemble: `raw_xsz6__signed_ridge_a01__time90_a0.25`.

This folder now treats the current strict three-`ML_single` ensemble as the
active ML ensemble. It replaces the older expanded-history ensemble line as the
documented retained model in this archive.

## Scope

The ensemble uses only the three retained strict single ML models:

| Component | 2020 Pooled IC | 2020 Monthly Mean |
| --- | ---: | ---: |
| MLP `time120_slope_a025_strong` | 0.050756 | 0.052954 |
| LGB `ref_time90_a1_signed_abs12_a08` | 0.050034 | 0.052653 |
| Ridge `simplex_basic_full2019` | 0.042481 | 0.044379 |

The selected strict candidate uses six views:

```text
mlp, lgb, ridge, mlp_xsz, lgb_xsz, ridge_xsz
```

where `*_xsz` is the same-timestamp cross-sectional z-score view of each model's
prediction.

## Architecture

The active ensemble is a small, auditable post-model stack:

1. Build rolling train-before-test predictions from the three single models.
2. Add cross-sectional views for each component, including `xcenter`, `xsz`,
   `xrank`, rank-gaussian, and tanh-z variants.
3. Screen 137 classic ensemble candidates on 2019 outer folds:
   equal weights, top-1, positive-IC weights, nonnegative simplex, signed ridge,
   and conservative time-bucket post-calibration.
4. Select by 2019-only selectors. The retained candidate wins the mean/std,
   q3+h2, q3+q4+h2, min+mean, and h2 selectors.
5. Fit on full 2019 and audit once on 2020.

The retained candidate is:

```text
raw_xsz6__signed_ridge_a01__time90_a0.25
```

Interpretation:

- `raw_xsz6`: use raw plus cross-sectional z-score views from MLP, LGB, and
  Ridge.
- `signed_ridge_a01`: fit a small signed ridge stack with `alpha=0.1`.
- `time90_a0.25`: apply a conservative 90-minute intraday time-bucket
  multiplier with strength `0.25`.

## Performance

| Model | 2020 Pooled IC | 2020 Monthly Mean | 2020 Monthly IR |
| --- | ---: | ---: | ---: |
| MLP single | 0.050756 | 0.052954 | n/a |
| LGB single | 0.050034 | 0.052653 | n/a |
| Ridge single | 0.042481 | 0.044379 | n/a |
| `raw_xsz6__signed_ridge_a01__time90_a0.25` | 0.057293 | 0.059619 | 4.9546 |

The strict retained ensemble improves pooled IC by about `+12.9%` versus the
best single model (`0.057293 / 0.050756 - 1`).

For context, a non-selected 2020 diagnostic candidate,
`mlp_lgb_raw2__signed_ridge_a1__time90_a0.25`, reached pooled IC `0.057845`,
but it is not the retained strict model because it was not selected by the 2019
selectors.

## Current Rebuild Status

The full rebuild requires large intermediate prediction parquet files under
`/root/autodl-tmp/quant/ML/effective_rolling_results` and
`/root/autodl-tmp/quant/ML/agent_runs`. Those source artifacts are not all
present on this machine, so `model/three_model_ensemble.py` materializes the
archived strict result and records the missing inputs in
`configs/required_inputs.csv`.

## Files

| Path | Purpose |
| --- | --- |
| `model/three_model_ensemble.py` | Active retained ensemble scaffold and input audit. |
| `scripts/run_best_ensemble.py` | Materializes the retained strict ensemble artifacts. |
| `configs/selected_by_2019.json` | 2019-only selected strict candidate. |
| `configs/best_2020_diagnostic.json` | Best 2020 diagnostic candidate, not retained. |
| `configs/candidate_catalog.csv` | 137-candidate classic ensemble search space. |
| `configs/required_inputs.csv` | Required large source artifacts and availability. |
| `metrics/best_ensemble_audit_metrics.csv` | Active strict ensemble 2020 audit metrics. |
| `metrics/selector_winners_2020_audit.csv` | 2019 selector winners and 2020 audit. |
| `metrics/single_model_audit_metrics.csv` | Three single-model baseline audits. |
| `weights/selected_outer_fold_weights.csv` | Archived 2019 outer-fold weights for the retained candidate. |
| `figures/three_model_ensemble_comparison.png` | Pooled IC comparison chart. |

## Reproduce

```bash
cd /root/jump_model/ML_ensemble
python scripts/run_best_ensemble.py
python model/three_model_ensemble.py --check-inputs
```
