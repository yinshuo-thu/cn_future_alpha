# v3: FactorOperatorBank + Time-Biased Market-Gated Multi-Scale Transformer

This folder archives the retained v7 large end-to-end Transformer. It keeps the
v2 TimeBias + MarketGate + stable residual backbone and adds a small, gated
online factor operator bank. Among the v7 ideas tested, this was the only one
that improved both the primary pooled IC and the stricter sector-neutral
non-overlap IC versus v2.

## Method Framework

![V3 FactorOperatorBank framework](figures/factor_operator_bank_time_biased_market_gated_transformer_framework.png)

The image above was generated from the v2 framework and updated with the
retained v3 optimization: `FactorOperatorBank`.

## What Changed From v2

v3 adds an online operator bank before the market-gated feature mixer:

- Enable `FactorOperatorBank`.
- Use rolling windows `[3, 5, 8, 13, 21, 34, 55, 89]`.
- Build 483 online operators:
  - 19 base operators;
  - 53 operators per rolling window;
  - 40 short/long pair-window difference operators.
- Gate operators from sequence-level summaries: latest value, mean, and
  standard deviation over the full input window.
- Keep `top_k=96` active operators.
- Use `project` output mode, producing a compact 96-dimensional factor tensor.
- Inject the factor tensor with `factor_scale_init=-2.0`, so the added branch
  starts with a small scale, roughly `sigmoid(-2.0) = 0.119`.

The rest of the retained backbone remains the v2 shape:
MarketGatedFeatureMixer, causal convolution stem, multi-scale patch embedding,
TimeBias attention, SwiGLU, LayerScale residuals, attention pooling,
last-by-scale pooling, and a six-target multitask head.

## FactorOperatorBank Intuition

The original 33 inputs are strong normalized primitives, but they leave many
microstructure combinations for the Transformer to rediscover. The factor bank
creates a controlled library of useful nonlinear operators at forward time,
then lets a small gate choose a compact subset per sequence.

Operator examples include:

- return, open-close, high-low, upper/lower shadow and close-position proxies;
- return times volume, amount, and open-interest changes;
- rolling momentum, realized volatility, upside/downside volatility;
- rolling z-scores and Bollinger-style distance;
- return-volume, return-amount, and return-OI correlations;
- stochastic and MFI-like signed-volume pressure proxies;
- short-window minus long-window trend differences.

The key design choice is restraint. The first attempt used `top_k=160` without
a small initial factor scale and underperformed. The retained v3 version uses
`top_k=96` plus the small residual scale, so the bank can add signal without
overpowering the v2 path early in training.

## Architecture

The active model is still `E2E_GatedMSPatch_MTL_DataLimited_v46`, with the
factor branch enabled.

| Field | Value |
| --- | ---: |
| Input window | 240 one-minute bars |
| Input features | 33 |
| Factor operators | 483 |
| Active factor operators | 96 |
| Hidden width | 192 |
| Transformer layers | 5 |
| Attention heads | 6 |
| SwiGLU hidden | 512 |
| Patch scales | `(4,2)`, `(8,4)`, `(16,8)`, `(32,16)` |
| Config parameter count | 5,231,687 |

Pipeline:

1. Normalize the 240 x 33 input sequence.
2. Build online factor operators from the normalized sequence.
3. Score operators using sequence summaries and keep the top 96.
4. Project and scale the selected operators, then concatenate them with the
   original features.
5. Feed `[raw features, scaled factor features]` into the market-gated feature
   mixer.
6. Continue through the v2 causal-conv, patching, TimeBias Transformer, dual
   pooling, and multitask head.

## Weight Files

The `weights/` directory contains raw PyTorch state dicts:

| Path | Meaning |
| --- | --- |
| `weights/model_raw.pt` | Final raw checkpoint for the retained v3 validation run. |
| `weights/snapshot_epoch1.pt` | Epoch-1 snapshot from the same retained shape. |

The state dict has 132 tensor entries. Important v3-specific keys include:

- `factor_scale`
- `factor_bank.gate.*`
- `factor_bank.project.weight`
- `factor_bank.out_norm.*`

These keys are absent in v2. The added tensors account for the v3 parameter
increase from about 5.03M to 5.23M.

## Metrics

2019 validation uses train `[2017-01-01, 2019-01-01)` and eval
`[2019-01-01, 2020-01-01)`.

| Metric | 2019 Validation | 2020 OOS |
| --- | ---: | ---: |
| Pooled IC | 0.064172 | 0.054808 |
| SN non-overlap IC | 0.070858 | 0.061614 |
| Raw non-overlap IC | 0.080524 | 0.067706 |
| Dense IC | 0.080186 | 0.074791 |
| Merged IC | 0.063401 | 0.054592 |
| SN RankIC | 0.080474 | 0.072599 |
| Scored rows | 2,693,337 | 2,827,738 |

Relative to v2 validation, v3 improved pooled IC by `+0.002103` and
SN non-overlap IC by `+0.000995`. On the 2020 OOS refit/eval, v3 improved
pooled IC by `+0.006649` and SN non-overlap IC by `+0.000249`.

## Ideas Tested But Not Retained

The v7 validation report kept only ideas that improved both decision metrics.

| Attempt | Pooled IC | SN non-overlap IC | Decision |
| --- | ---: | ---: | --- |
| v2 baseline market-only | 0.062069 | 0.069863 | baseline |
| Factor bank `top_k=160` | 0.059374 | 0.067541 | rejected |
| Factor bank `top_k=96`, scaled project | 0.064172 | 0.070858 | retained |
| Low-rank input replacement | 0.054888 | 0.064736 | rejected |
| Low-rank residual | 0.064778 | 0.069182 | rejected; SN down |
| Full metadata embedding | 0.055533 | 0.067231 | rejected |
| Symbol + minute metadata | 0.060745 | 0.070177 | rejected |
| 4-expert MoE head | 0.058668 | 0.067202 | rejected |
| 2-expert MoE head | 0.060294 | 0.064352 | rejected |

## Files

| Path | Content |
| --- | --- |
| `src/modelv46.py` | Active implementation with FactorOperatorBank and optional branches. |
| `scripts/run_transformer_v4_v6.py` | Training/evaluation entrypoint snapshot. |
| `configs/run_config.json` | 2019 validation run config. |
| `configs/run_config_2020_oos.json` | 2020 OOS refit/eval config. |
| `metrics/` | Compact validation and OOS metric artifacts. |
| `reports/23_v7_ideas_validation_summary.md` | v7 idea-selection report. |
| `reports/24_v7_factor_bank_k96_scaled_project_2020_oos_final.md` | 2020 OOS report. |
| `weights/` | Archived PyTorch state dicts. |
| `figures/` | Generated framework diagram for the retained v3 shape. |

## Reproduce

The archived scripts keep the original absolute project root:
`/root/autodl-tmp/quant/end2end_30m`.

```bash
cd /root/autodl-tmp/quant/end2end_30m
python scripts/run_transformer_v4_v6.py \
  --run-name transformer_v7_factor_bank_k96_scaled_project_e1 \
  --train-start 2017-01-01 --train-end 2019-01-01 \
  --eval-start 2019-01-01 --eval-end 2020-01-01 \
  --disable-layer-fusion --disable-revin --disable-cross-section \
  --disable-cross-variate \
  --use-factor-bank --factor-top-k 96 \
  --factor-output-mode project --factor-scale-init -2.0
```

