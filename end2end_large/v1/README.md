# v1: Gated Multi-Scale Patch Transformer With Dual Pooling

This folder archives the first strong large end-to-end Transformer baseline in
the current model ladder.

## Method Framework

![Gated Multi-Scale Patch Transformer with Dual Pooling](<Gated Multi-Scale Patch Transformer with Dual Pooling.png>)

## Structure

The model reads 33 normalized 1-minute bar/factor features over a 240-minute
window and predicts six multitask labels, with the 30-minute proxy return target
as the main IC target.

Core blocks:

- `GatedFeatureMixer`: maps raw features into `d_model=192` with a learned
  feature gate.
- `CausalConvStem`: adds short-range causal temporal smoothing before patching.
- `MultiScalePatchEmbedding`: builds patch tokens at `(4,2)`, `(8,4)`,
  `(16,8)`, and `(32,16)` patch/stride scales.
- Five pre-norm Transformer blocks with six attention heads.
- Dual pooling:
  - attention pooling over all patch tokens;
  - last-token pooling per scale to preserve recency.
- `MultiTaskHead`: shared representation with six supervised output targets.

## Archived Files

| Path | Content |
| --- | --- |
| `src/` | Source snapshot for model, metrics, data, normalization, and training helpers. |
| `scripts/run_transformer_main_config.py` | Original training/evaluation entrypoint. |
| `weights/model.pt` | Trained v1 validation checkpoint. |
| `configs/run_config.json` | Run metadata and hyperparameters. |
| `metrics/` | Compact 2019 validation metrics. Prediction parquet files are intentionally excluded. |
| `reports/11_transformer_main_vs_ridge_gap.md` | Original experiment report snippet. |

## 2019 Validation Metrics

| Metric | Value |
| --- | ---: |
| Pooled IC | 0.054609 |
| SN non-overlap IC | 0.064411 |
| Raw non-overlap IC | 0.070722 |
| Dense IC | 0.070378 |
| Merged IC | 0.054452 |
| SN RankIC | 0.073766 |
| Parameters | 5,503,433 |

The root README reports the separate 2020 OOS test performance used for the
cross-family retained-results table.

## Reproduce

```bash
cd /root/autodl-tmp/quant/end2end_30m
python scripts/run_transformer_main_config.py \
  --run-name transformer_main_config_2019_full \
  --train-start 2017-01-01 --train-end 2019-01-01 \
  --eval-start 2019-01-01 --eval-end 2020-01-01
```
