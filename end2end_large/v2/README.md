# v2: Time-Biased Market-Gated Multi-Scale Transformer With Stable Residual Learning

This folder archives the retained v6 large end-to-end Transformer. It is the
first version in this ladder that clearly improved the v1 pooled IC and
sector-neutral non-overlap IC under the strict 2019 validation gate.

## Method Framework

![Time-Biased Market-Gated Multi-Scale Transformer with Stable Residual Learning](<Time-Biased Market-Gated Multi-Scale Transformer with Stable Residual Learning.png>)

## What Changed From v1

v2 keeps the useful v1 skeleton: 240 one-minute bars, 33 normalized features,
multi-scale patch tokens, dual readout, and a six-target multitask head. The
retained changes are:

- `MarketGatedFeatureMixer`: conditions each symbol's feature mix on the
  same-timestamp market state, computed as cross-symbol mean and standard
  deviation of the latest normalized features.
- `TimeBiasSelfAttention`: adds a per-head temporal distance decay to attention
  logits, so nearer patch tokens receive an explicit recency prior.
- `SwiGLU` feed-forward blocks: use a gated nonlinear value path in each
  Transformer block.
- `LayerScale` residuals: learn small residual scales on attention and FFN
  branches, making the block updates more stable.
- Removed rejected branches: layer fusion, RevIN, cross-section attention, and
  cross-variate attention were tested and not retained.

## Architecture

The active model is `E2E_GatedMSPatch_MTL_DataLimited_v46` with:

| Field | Value |
| --- | ---: |
| Input window | 240 one-minute bars |
| Input features | 33 |
| Hidden width | 192 |
| Transformer layers | 5 |
| Attention heads | 6 |
| SwiGLU hidden | 512 |
| Patch scales | `(4,2)`, `(8,4)`, `(16,8)`, `(32,16)` |
| Dropout / attention dropout / feature dropout | `0.15 / 0.10 / 0.05` |
| Config parameter count | 5,026,637 |

Pipeline:

1. Normalize raw input and build same-timestamp market state.
2. Combine a base linear branch and a nonlinear mix branch with a learned
   market gate.
3. Apply a causal convolution stem for short-range local smoothing.
4. Convert the sequence into multi-scale patch tokens and attach patch time
   coordinates.
5. Run five TimeBias Transformer blocks:
   `LN -> QKV -> TimeBiasAttention -> LayerScale`, then
   `LN -> SwiGLU -> LayerScale`.
6. Read out with attention pooling and last-token-by-scale pooling, then predict
   six multitask labels.

The primary score used in the retained tables is the raw pooled IC of the
30-minute proxy return. The internal stability diagnostic is sector-neutral
non-overlap cross-sectional IC.

## Weight Files

The `weights/` directory contains raw PyTorch state dicts:

| Path | Meaning |
| --- | --- |
| `weights/model_raw.pt` | Final raw checkpoint used for the archived v2 validation run. |
| `weights/snapshot_epoch1.pt` | Epoch-1 snapshot from the same retained shape. |

The state dict has 122 tensor entries. Important signs that this is the v2
shape:

- `mixer.gate.weight` and `mixer.gate.bias` are present, confirming the
  market-gated feature mixer.
- `blocks.*.gamma_attn` and `blocks.*.gamma_ffn` are present, confirming
  LayerScale residuals.
- No `factor_bank.*` keys are present; the online factor bank is introduced in
  v3.

## Metrics

2019 validation uses train `[2017-01-01, 2019-01-01)` and eval
`[2019-01-01, 2020-01-01)`.

| Metric | 2019 Validation | 2020 OOS |
| --- | ---: | ---: |
| Pooled IC | 0.062069 | 0.048159 |
| SN non-overlap IC | 0.069863 | 0.061365 |
| Raw non-overlap IC | 0.078247 | 0.064287 |
| Dense IC | 0.077234 | 0.070408 |
| Merged IC | 0.061555 | 0.048316 |
| SN RankIC | 0.079789 | 0.067758 |
| Scored rows | 2,693,337 | 2,827,738 |

The original 2019 and 2020 reports are in `reports/16_*.md`,
`reports/17_*.md`, and `reports/18_*.md`.

## Files

| Path | Content |
| --- | --- |
| `src/modelv46.py` | Active v2/v3 model implementation with optional branches. |
| `scripts/run_transformer_v4_v6.py` | Training/evaluation entrypoint snapshot. |
| `configs/run_config.json` | 2019 validation run config. |
| `configs/run_config_2020_oos.json` | 2020 OOS refit/eval config. |
| `metrics/` | Compact validation and OOS metric artifacts. |
| `weights/` | Archived PyTorch state dicts. |

## Reproduce

The archived scripts keep the original absolute project root:
`/root/autodl-tmp/quant/end2end_30m`.

```bash
cd /root/autodl-tmp/quant/end2end_30m
python scripts/run_transformer_v4_v6.py \
  --run-name transformer_v6_2019_validation_market_only_from_v4_nolf_e1 \
  --train-start 2017-01-01 --train-end 2019-01-01 \
  --eval-start 2019-01-01 --eval-end 2020-01-01 \
  --disable-layer-fusion --disable-revin --disable-cross-section \
  --disable-cross-variate
```

