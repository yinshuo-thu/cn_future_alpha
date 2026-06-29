# End-to-End Large Model Evolution

This directory archives the three large end-to-end Transformer iterations that
were retained as the main model ladder for the China futures 30-minute return
task. The three folders are intentionally named `v1`, `v2`, and `v3` to reflect
the evolution requested for this archive.

The primary project metric from the Jump PDF is **Pooled IC**: flatten
predictions and labels across the chosen time period and all 51 symbols, then
compute the cosine-style IC

```text
IC = mean(alpha * label) / sqrt(mean(alpha * alpha) * mean(label * label))
```

`SN non-overlap IC` is a stricter internal diagnostic: sector-neutral
cross-sectional Pearson IC on stride-30 timestamps. It is useful for stability
and leakage checks, but the PDF threshold of `0.05` refers to the pooled IC.

## Archived Folders

| Folder | Model Name | Source Run | Main Weight |
| --- | --- | --- | --- |
| `v1/` | Gated Multi-Scale Patch Transformer with Dual Pooling | `transformer_main_config_2019_full` | `weights/model.pt` |
| `v2/` | Time-Biased Market-Gated Multi-Scale Patch Transformer with Stable Residual Learning | `transformer_v6_2019_validation_market_only_from_v4_nolf_e1` | `weights/model_raw.pt` |
| `v3/` | FactorOperatorBank + Time-Biased Market-Gated Multi-Scale Patch Transformer | `transformer_v7_factor_bank_k96_scaled_project_e1` | `weights/model_raw.pt` |

Each folder contains:

- `src/`: model and evaluation source snapshot.
- `scripts/`: training/evaluation entrypoint used by that branch.
- `weights/`: trained PyTorch state dicts.
- `configs/`: run config plus small data/sector metadata.
- `metrics/`: compact CSV metric artifacts, excluding prediction parquet files.
- `reports/`: original experiment report snippets where available.
- `figures/`: architecture diagrams where available.

## 2019 Validation Comparison

All three rows below use the same validation protocol:

- train: `[2017-01-01, 2019-01-01)`
- eval: `[2019-01-01, 2020-01-01)`
- scored rows: `2,693,337`
- sequence length: `240`
- input feature count: `33`
- targets: six multitask labels, with `proxy_ret_30m_cs_zscore` as the main training target

| Version | Parameters | Pooled IC | SN non-overlap IC | Raw non-overlap IC | Dense IC | Merged IC | SN RankIC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| v1 | 5,503,433 | 0.054609 | 0.064411 | 0.070722 | 0.070378 | 0.054452 | 0.073766 |
| v2 | 5,026,637 | 0.062069 | 0.069863 | 0.078247 | 0.077234 | 0.061555 | 0.079789 |
| v3 | 5,231,687 | 0.064172 | 0.070858 | 0.080524 | 0.080186 | 0.063401 | 0.080474 |

Relative to v1, v3 improves pooled IC by `+0.009563` and SN non-overlap IC by
`+0.006447`. Relative to v2, v3 improves pooled IC by `+0.002103` and SN
non-overlap IC by `+0.000995`.

## v1: Gated Multi-Scale Patch Transformer With Dual Pooling

The v1 branch is the first strong deep end-to-end baseline.

Core structure:

- `GatedFeatureMixer`: maps the 33 normalized bar/factor features into
  `d_model=192` with a learned gate.
- `CausalConvStem`: local causal temporal smoothing before patching.
- Multi-scale patch embedding with patch/stride pairs
  `(4,2), (8,4), (16,8), (32,16)`.
- Five pre-norm Transformer blocks with six heads.
- Dual pooling:
  - attention pooling over all patch tokens;
  - last-token pooling per scale.
- Multitask MLP head for 5m/10m/30m/60m return plus volatility/range auxiliary
  labels.

Useful intuition:

- Multi-scale patching follows the same broad idea as PatchTST-style time-series
  patch tokens: shorter patches keep local microstructure, longer patches
  summarize slower intraday state.
- The gated feature mixer is a lightweight channel mixer inspired by gated/GLU
  feature selection: useful when many normalized inputs are weak individually.
- Dual pooling was effective because attention pooling captures distributed
  evidence while last-by-scale pooling preserves recency.

## v2: Time-Biased Market-Gated Stable Residual Transformer

The v2 branch starts from v1 and keeps only the v4-v6 changes that survived
2019 validation.

Effective modifications:

- **TimeBiasAttention**: attention logits receive a real-time-distance decay
  bias. This is related to ALiBi (Press et al., 2022), but uses actual patch
  time coordinates rather than token index distance.
- **SwiGLU feed-forward blocks**: inspired by GLU/SwiGLU work (Shazeer, 2020)
  and common modern Transformer practice.
- **LayerScale residuals**: small learnable residual scales, related to CaiT /
  LayerScale (Touvron et al., 2021), made the deeper residual path calmer.
- **MarketGatedFeatureMixer**: augments each sample with same-timestamp market
  state (`mean` and `std` across symbols) and gates the per-symbol feature mix.
  This is a lightweight cross-sectional regime conditioning mechanism without
  full cross-sectional attention.

Why it helped:

- Time bias matched the recency nature of intraday futures signals.
- Stable residual learning reduced sensitivity to Transformer block changes.
- Market gating gave the model a condition on broad market state, which is
  often decisive for whether a local pattern means continuation or reversal.

Related sources:

- ALiBi: Press, Smith, and Lewis, "Train Short, Test Long: Attention with Linear
  Biases Enables Input Length Extrapolation", ICLR 2022.
- SwiGLU / GLU variants: Shazeer, "GLU Variants Improve Transformer", 2020.
- LayerScale: Touvron et al., "Going deeper with Image Transformers", ICCV 2021.

Prior v4-v6 ablations found that layer fusion, RevIN, cross-sectional attention,
and cross-variate attention were not retained for this data/validation setup.
The final v2 shape is therefore the market-only version with TimeBias,
SwiGLU/LayerScale, and market gating.

## v3: FactorOperatorBank On Top Of v2

The v3 branch is the latest retained model. It keeps v2 and adds an online
`FactorOperatorBank`.

Effective modification:

- The factor bank constructs 483 operators inside the network at forward time.
- It uses windows `[3, 5, 8, 13, 21, 34, 55, 89]`.
- Decomposition:
  - 19 global operators;
  - 53 operators per rolling window;
  - 40 short/long trend-difference pair operators.
- Operators include K-line body/shadow proxies, return times volume/amount/OI
  changes, rolling momentum, realized volatility, upside/downside volatility,
  volume-price correlation, OI momentum, stochastic/Bollinger-style proxies,
  MFI-style signed-volume proxy, and short-long trend differences.
- A sequence gate reads last/mean/std over the full input window and selects a
  limited active bank. The retained version uses `top_k=96`.
- The selected factor bank is projected and injected with
  `factor_scale_init=-2.0`, so the model starts by using the new factors gently
  and learns whether to increase their impact.

Why it helped:

- The raw 33 inputs already contain useful rolling-normalized primitives, but
  the bank gives the Transformer richer nonlinear operator families without
  committing to a huge static feature table.
- The first attempt with `top_k=160` and no small initial scale overpowered the
  existing signal path and underperformed.
- The retained `top_k=96` scaled design improved both the PDF primary Pooled IC
  and the stricter SN non-overlap IC.

Related inspiration:

- Classical technical-analysis operator families: momentum, realized
  volatility, Bollinger z-score, stochastic oscillator, MFI-like signed-volume
  pressure.
- Neural feature selection/gating: the gate is used as a compact, sequence-level
  operator selector rather than as an unconstrained feature expansion.
- Factorization-machine intuition is adjacent: useful financial signals often
  emerge from combinations such as return times volume change or volatility
  conditioned on OI change. In this run, explicit low-rank pairwise interaction
  was tested separately and not retained.

## Attempts Not Retained

The following ideas were each tested at least twice when the first design failed
or looked ambiguous:

| Attempt | Best 2019 Pooled IC | Best 2019 SN non-overlap IC | Decision |
| --- | ---: | ---: | --- |
| FactorOperatorBank `top_k=160`, no small scale | 0.059374 | 0.067541 | rejected; too disruptive |
| LowRankFeatureInteraction, replacement input block | 0.054888 | 0.064736 | rejected |
| LowRankFeatureInteraction, residual small-scale branch | 0.064778 | 0.069182 | rejected; pooled up but SN down |
| Full metadata embeddings: symbol, minute, day, month | 0.055533 | 0.067231 | rejected |
| Conservative metadata: symbol + minute only | 0.060745 | 0.070177 | rejected; did not beat v3 |
| MoE head: four experts + balance penalty | 0.058668 | 0.067202 | rejected |
| MoE head: two experts, no balance penalty | 0.060294 | 0.064352 | rejected |

These negative results are included because they were useful guardrails: the
model benefited from richer online operators, but not from making the final
predictor or metadata conditioning more fragmented under this 2019 validation
gate.

## Reproduction Notes

The archived scripts are faithful snapshots from the original experiment tree.
They keep the original absolute `PROJECT_ROOT` pointing at
`/root/autodl-tmp/quant/end2end_30m`. For a fresh machine, either restore the
same path or update `PROJECT_ROOT` and the `panel_cache` path in
`configs/data_limited_panel_thr12.json`.

Example commands from the original environment:

```bash
# v1
python scripts/run_transformer_main_config.py \
  --run-name transformer_main_config_2019_full \
  --train-start 2017-01-01 --train-end 2019-01-01 \
  --eval-start 2019-01-01 --eval-end 2020-01-01

# v2
python scripts/run_transformer_v4_v6.py \
  --run-name transformer_v6_2019_validation_market_only_from_v4_nolf_e1 \
  --version v6 \
  --train-start 2017-01-01 --train-end 2019-01-01 \
  --eval-start 2019-01-01 --eval-end 2020-01-01 \
  --disable-layer-fusion --disable-revin --disable-cross-section \
  --disable-cross-variate --no-eval-ema

# v3
python scripts/run_transformer_v4_v6.py \
  --run-name transformer_v7_factor_bank_k96_scaled_project_e1 \
  --version v6 \
  --train-start 2017-01-01 --train-end 2019-01-01 \
  --eval-start 2019-01-01 --eval-end 2020-01-01 \
  --disable-layer-fusion --disable-revin --disable-cross-section \
  --disable-cross-variate \
  --use-factor-bank --factor-top-k 96 --factor-output-mode project \
  --factor-scale-init -2.0 --no-eval-ema
```

No raw CSVs, panel parquet files, or prediction parquet files are included in
this archive. The `.pt` files are trained model weights and are intentionally
included.

## Weight Checksums

See `weights_sha256.txt` for SHA-256 checksums of the archived final weights.
