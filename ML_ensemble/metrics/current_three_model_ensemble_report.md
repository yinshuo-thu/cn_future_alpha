# Current Three-Model Ensemble, 2026-06-29

## Scope

Components are the current strict single-model lines:

| component | 2020 pooled IC | 2020 monthly mean |
|---|---:|---:|
| MLP `time120_slope_a025_strong` | 0.0507556821 | 0.0529536228 |
| LGB `ref_time90_a1_then_signed_abs12_a0.8` | 0.0500341707 | 0.0526525597 |
| Ridge `simplex_basic_full2019_to_2020` | 0.0424810000 | 0.0443790000 |

All strict selectors use only 2019 outer folds. 2020 is used only after the candidate/selector is fixed.

## Best Results

| status | candidate | 2019 selector | 2020 pooled IC | 2020 monthly mean |
|---|---|---|---:|---:|
| strict selector winner | `raw_xsz6__signed_ridge_a01__time90_a0.25` | mean/std, q3+h2, q3+q4+h2, min+mean, h2 | 0.057293 | 0.059619 |
| strict selector winner | `mlp_lgb_xsz2__equal` | q4 | 0.055013 | 0.057297 |
| best 2020 diagnostic | `mlp_lgb_raw2__signed_ridge_a1__time90_a0.25` | not selected by 2019 selectors | 0.057845 | 0.060260 |
| best simple diagnostic | `mlp_lgb_raw2__equal` | not selected by 2019 selectors | 0.057767 | 0.060241 |

Interpretation: the ensemble helps a lot versus any one of the three current singles, but the best lift comes mostly from MLP + LGB. Ridge receives little or no useful weight in the strongest candidates.

## Previous Ensemble Comparison

| benchmark | 2020 pooled IC | 2020 monthly mean | result |
|---|---:|---:|---|
| historical `predictions_best_ic0716` pred | 0.069877 | 0.071980 | not beaten |
| expanded clean stack | 0.059138 | 0.061730 | not beaten |
| historical `predictions_best_ic0716` xsz | 0.058103 | 0.060247 | not beaten |
| core MoE no-DL pred | 0.056150 | 0.058712 | beaten by strict and diagnostic |

## Files

- `ensemble_2019_cv_screen.csv`: all 2019 CV candidate scores.
- `ensemble_2020_audit_selected.csv`: selected/top candidate 2020 audits.
- `ensemble_selector_winners_2020_audit.csv`: strict selector winners only.
- `ensemble_final_weights.csv`: final full-2019 fitted weights.
- `previous_ensemble_benchmarks.csv`: recomputed/loaded previous benchmark comparison.

