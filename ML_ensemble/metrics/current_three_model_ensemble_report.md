# Retained Three-Model Ensemble, 2026-06-29

## Scope

Components are the current strict single-model lines:

| Component | 2020 pooled IC | 2020 monthly mean |
|---|---:|---:|
| MLP `time120_slope_a025_strong` | 0.050756 | 0.052954 |
| LGB `ref_time90_a1_signed_abs12_a08` | 0.050034 | 0.052653 |
| Ridge `simplex_basic_full2019` | 0.042481 | 0.044379 |

All selectors use only 2019 outer folds. 2020 is used only after the
candidate/selector is fixed.

## Retained Model

| Status | Candidate | 2019 selector | 2020 pooled IC | 2020 monthly mean | 2020 monthly IR |
|---|---|---|---:|---:|---:|
| retained strict selector winner | `raw_xsz6__signed_ridge_a01__time90_a0.25` | mean/std, q3+h2, q3+q4+h2, min+mean, h2 | 0.057293 | 0.059619 | 4.9546 |

Interpretation: the retained ensemble improves pooled IC by about `+12.9%`
versus the best current single model. It uses raw and cross-sectional z-score
views from MLP, LGB, and Ridge, then fits a small signed ridge stack with a
conservative 90-minute time-bucket calibration.

## Diagnostic Context

| Candidate | Selection status | 2020 pooled IC | 2020 monthly mean |
|---|---|---:|---:|
| `mlp_lgb_raw2__signed_ridge_a1__time90_a0.25` | best 2020 diagnostic, not selected by 2019 selectors | 0.057845 | 0.060260 |
| `mlp_lgb_raw2__equal` | best simple diagnostic, not selected by 2019 selectors | 0.057767 | 0.060241 |
| `mlp_lgb_xsz2__equal` | q4 selector winner | 0.055013 | 0.057297 |

The diagnostic rows are kept for audit context only. The active replacement for
`ML_ensemble` is `raw_xsz6__signed_ridge_a01__time90_a0.25`.

## Files

- `../configs/candidate_catalog.csv`: 137-candidate classic ensemble search
  space.
- `best_ensemble_audit_metrics.csv`: retained strict ensemble 2020 audit.
- `best_diagnostic_audit_metrics.csv`: best 2020 diagnostic row, not retained.
- `selector_winners_2020_audit.csv`: 2019 selector winners audited once on 2020.
- `single_model_audit_metrics.csv`: MLP/LGB/Ridge baseline audits.
- `../weights/selected_outer_fold_weights.csv`: archived 2019 outer-fold weights
  for the retained candidate.
