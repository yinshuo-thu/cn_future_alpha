# End-to-End Single

Retained run:
`factor_operator_market_full_oldarch_fullsample_m05_m12_raw_seed44`.

This is a small-scale feasibility model. It tests whether raw 1-minute market
channels can be converted into differentiable factor-like operator responses,
then scaled through trainable gates, attention, and a compact MoE head.

## Architecture

| Block | Core Idea | Size / Parameters |
| --- | --- | --- |
| Raw input | 35 raw/market channels from history-only normalized 1-minute bars, sequence length 120. | 35 input features |
| Factor operator bank | Differentiable rolling/operator features over windows `[3,5,8,13,21,34,55,89]`; extra pair/global ops enabled. | 483 operator responses |
| Operator gate | Sequence gate uses last/mean/std state to activate operators per sample. | hidden 80, top-48 |
| Projection | Operator responses projected and concatenated with raw features. | 96 projection dim |
| Interaction block | Low-rank FM-style feature interaction before temporal encoding. | interaction dim 128, rank 8 |
| Temporal model | Conv1D patch embedding, PatchTST-style pre-norm Transformer. | d_model 128, 4 heads, 4 layers, ffn 256 |
| Pool/head | Attention pooling plus 4-expert MoE prediction head. | 4 experts, hidden 64 |
| Total | State-dict parameter count from checkpoint. | 1,421,371 params |

## Metrics

| Metric | Value |
| --- | ---: |
| Pooled IC | 0.039660 |
| SN non-overlap IC | 0.031582 |
| Merged RankIC | 0.049840 |
| Monthly ICIR | 2.280238 |
| Scored rows | 2,213,665 |

## Why Keep It

- It proves the causal factor-operator idea is viable without precomputed ML
  factor panels.
- It shows that hand-designed factor processes can become trainable weights.
- It provides the bridge idea used later by the larger branches: candidate
  factor/operator responses can be gated, selected, or softly weighted.
- It stays compact enough to audit and retrain quickly.

## Files

| Path | Purpose |
| --- | --- |
| `src/` | End-to-end data, rolling, model, train, validation, and visualization code. |
| `configs/factor_operator_market_full_oldarch_fullsample_m05_m12_raw_seed44.yaml` | Exact run config. |
| `weights/2020M05.pt` ... `weights/2020M12.pt` | Trained monthly checkpoints. |
| `metrics/audit_metrics.csv` | Same-standard pooled/SN audit metrics. |
| `figures/single_dashboard.png` | Monthly IC and 20-bin return dashboard. |

Train from a copied config after setting raw-data/cache paths:

```bash
cd /root/jump_model/end2end_single
python -m src.train --config configs/factor_operator_market_full_oldarch_fullsample_m05_m12_raw_seed44.yaml
```
