# cn_future_alpha

This archive keeps the useful no-future-leakage infrastructure, model code,
small audit artifacts, dashboards, and trained weights needed to rebuild the
project from raw high-frequency China futures data.

Raw data is intentionally not included. Download it from:
https://www.kaggle.com/datasets/wentinglu/highfrequency-futures-data-china/data

## Layout

| Directory | Content |
| --- | --- |
| `ML_single/` | Three strict ML single models: MLP, LightGBM, Ridge. Includes factor/model infrastructure, selected configs, lightweight postprocess weights, metrics, and dashboards. |
| `ML_ensemble/` | Retained strict three-ML-single ensemble: `raw_xsz6__signed_ridge_a01__time90_a0.25`. |
| `end2end_single/` | Small-scale end-to-end feasibility model: `factor_operator_market_full_oldarch_fullsample_m05_m12_raw_seed44`. |
| `end2end_large/` | Three large end-to-end Transformer checkpoints organized as `v1` to `v3`: dual-pooling baseline, time-biased market-gated model, and the retained FactorOperatorBank model. |
| `tools/` | Audit generators for pooled IC, SN non-overlap IC, monthly IC, and 20-bin return plots. |
| `common_docs/` | Shared audit tables and data notes. |

## Metric Convention

`Pooled IC` is the flattened cosine-style IC specified by the Jump PDF: flatten
all predictions and labels across the chosen period and all symbols, then compute
`mean(alpha * label) / sqrt(mean(alpha^2) * mean(label^2))`. This is the metric
associated with the `0.05` reasonable-starting-point threshold. `SN non-overlap
IC` is an internal diagnostic: sector-neutral cross-sectional Pearson IC on
stride-30 timestamps. The retained ML ensemble is the strict 2019-selected
three-single-model stack with 2020 pooled IC `0.057293`.

## Retained Results

| Block | Model | Eval Window | Pooled IC | SN non-overlap IC |
| --- | --- | --- | ---: | ---: |
| ML single | `mlp_time120_slope_a025_strong` | 2020 | 0.050756 | 0.065097 |
| ML single | `lgb_ref_time90_a1_signed_abs12_a08` | 2020 | 0.050034 | 0.065138 |
| ML single | `ridge_simplex_basic_full2019` | 2020 | 0.042481 | 0.064183 |
| ML ensemble | `raw_xsz6__signed_ridge_a01__time90_a0.25` | 2020 | 0.057293 | n/a |
| end2end single | `factor_operator_market_full_oldarch_fullsample_m05_m12_raw_seed44` | 2020-05..2020-12 | 0.039660 | 0.031582 |
| end2end large v1 | `Gated Multi-Scale Patch Transformer with Dual Pooling` | 2020 | 0.043578 | 0.059084 |
| end2end large v2 | `Time-Biased Market-Gated Multi-Scale Patch Transformer with Stable Residual Learning` | 2020 | 0.048159 | 0.061365 |
| end2end large v3 | `FactorOperatorBank + Time-Biased Market-Gated Multi-Scale Patch Transformer` | 2020 | 0.054808 | 0.061614 |

`SN non-overlap IC` means sector-neutral cross-sectional Pearson IC computed on
stride-30 timestamps. It is a robustness/leakage diagnostic; `Pooled IC` is the
Jump PDF headline metric and the one associated with the `0.05` threshold.

## End-to-End Large Ladder

The current `end2end_large/` directory replaces the older exploratory large
branches with a cleaner three-step Transformer evolution:

1. `v1`: Gated multi-scale patch Transformer with causal stem and dual pooling.
2. `v2`: v1 plus TimeBiasAttention, SwiGLU/LayerScale stable residual learning,
   and market-state feature gating.
3. `v3`: v2 plus an online FactorOperatorBank with 483 constructed operators,
   sequence-level top-k gating (`top_k=96`), and small initial factor injection.

Only v3 is the latest retained large end-to-end model, and it remains the best
large Transformer on the 2020 pooled test after the 2019 validation selection.
Low-rank interaction, metadata embeddings, and MoE heads were tested and not
retained because they did not improve both Pooled IC and SN non-overlap IC. See
`end2end_large/README.md` for the detailed ablation record, architecture notes,
paper inspirations, weight paths, and checksums.

## No-Leakage Notes

All retained headline ML models use rolling train-before-test predictions. The
ensemble screens 137 classic three-model stack candidates on 2019 outer folds,
selects the strict winner before looking at 2020, refits on pre-2020 data, and
audits on 2020. `end2end_single/` keeps the older rolling-monthly feasibility
model.

The current `end2end_large/` v1-v3 ladder uses a fixed train-before-validation
selection split (`2017-2018` train, `2019` validation). The 2020 retained-result
rows then refit the selected Transformer shapes on pre-2020 data and audit them
on 2020. Features are causal, labels mask long-break horizons, and no validation
or test data is used for training. The Transformer ladder shows strong
performance persistence: IC levels naturally drop from validation to test, but
the model ordering is unchanged on both Pooled IC and SN non-overlap IC.

| Version | 2019 Val Pooled IC | 2020 Test Pooled IC | 2019 Val SN non-overlap IC | 2020 Test SN non-overlap IC |
| --- | ---: | ---: | ---: | ---: |
| v1 | 0.054609 | 0.043578 | 0.064411 | 0.059084 |
| v2 | 0.062069 | 0.048159 | 0.069863 | 0.061365 |
| v3 | 0.064172 | 0.054808 | 0.070858 | 0.061614 |

## Rebuild

Install the base environment:

```bash
cd /root/jump_model
pip install -r requirements.txt
```

Regenerate retained audit assets from local original experiment artifacts:

```bash
python ML_ensemble/scripts/run_best_ensemble.py
python tools/generate_end2end_audit_assets.py
```

No raw CSVs, factor panels, feature cache parquet files, or prediction parquet
files are included. Neural checkpoint files are retained because they are model
weights, not data.
