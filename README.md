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
| `ML_ensemble/` | Best strict ML ensemble: `expanded_gate_stack_2019q4_nonneg`. |
| `end2end_single/` | Small-scale end-to-end feasibility model: `factor_operator_market_full_oldarch_fullsample_m05_m12_raw_seed44`. |
| `end2end_large/` | Three useful larger end-to-end branches organized as `version1` to `version3`. |
| `tools/` | Audit generators for pooled IC, SN non-overlap IC, monthly IC, and 20-bin return plots. |
| `common_docs/` | Shared audit tables and data notes. |

## Metric Convention

`Pooled IC` is the flattened cosine-style IC used in the training/evaluation
scripts. `SN non-overlap IC` is sector-neutral cross-sectional Pearson IC on
stride-30 timestamps. The headline ML ensemble also has an original stack
summary IC of `0.059218`; the generic migration audit recomputation reports
`0.059138`, and both clear the `0.059` target.

## Retained Results

| Block | Model | Eval Window | Pooled IC | SN non-overlap IC |
| --- | --- | --- | ---: | ---: |
| ML single | `mlp_time120_slope_a025_strong` | 2020 | 0.050756 | 0.065097 |
| ML single | `lgb_ref_time90_a1_signed_abs12_a08` | 2020 | 0.050034 | 0.065138 |
| ML single | `ridge_simplex_basic_full2019` | 2020 | 0.042481 | 0.064183 |
| ML ensemble | `expanded_gate_stack_2019q4_nonneg` | 2020 | 0.059138 | 0.079266 |
| end2end single | `factor_operator_market_full_oldarch_fullsample_m05_m12_raw_seed44` | 2020-05..2020-12 | 0.039660 | 0.031582 |
| end2end large v1 | `factor_operator_extra_market_lite_seq_top48_huber_moe4_monthly12_val1` | 2020 | 0.027534 | 0.037190 |
| end2end large v2 | `factor_operator_extra_market_full_seq_top48_huber_moe4_monthly12_val1` | 2020 | 0.032294 | 0.036223 |
| end2end large v3 | `review_sttopk_xsz_dtmean_2019_2020_e3` | 2020 | 0.024410 | 0.037818 |

## No-Leakage Notes

All retained headline ML models use rolling train-before-test predictions. The
ensemble selects candidate gates by 2019Q4 validation, refits on pre-2020 data,
and audits on 2020. End-to-end models use rolling monthly splits where each test
month is predicted from prior windows only; normalization is history-only.

## Rebuild

Install the base environment:

```bash
cd /root/jump_model
pip install -r requirements.txt
```

Regenerate audit assets from local original experiment artifacts:

```bash
python tools/generate_ml_audit_assets.py
python tools/generate_end2end_audit_assets.py
```

No raw CSVs, factor panels, feature cache parquet files, or prediction parquet
files are included. Neural checkpoint files are retained because they are model
weights, not data.
