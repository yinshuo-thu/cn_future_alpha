# End-to-End Large Versions

This folder keeps three useful end-to-end branches from the later optimization
and ablation work. They are evolution stages rather than a monotonic
leaderboard.

## Version Table

| Version | Run | Main Idea | Params / Weights | Pooled IC | SN non-overlap IC |
| --- | --- | --- | --- | ---: | ---: |
| `version1` | `factor_operator_extra_market_lite_seq_top48_huber_moe4_monthly12_val1` | Lite factor-operator branch with extra market features, sequence top-48 operator gate, Huber loss, and a 4-expert MoE. | 1,401,511 params; 12 checkpoints, 64.7MB | 0.027534 | 0.037190 |
| `version2` | `factor_operator_extra_market_full_seq_top48_huber_moe4_monthly12_val1` | Full-market feature version of the same causal factor-operator/MoE design, kept as the stronger pooled-IC compact operator branch. | 1,421,371 params; 12 checkpoints, 65.6MB | 0.032294 | 0.036223 |
| `version3` | `review_sttopk_xsz_dtmean_2019_2020_e3` | Review-stage ST-top-k branch with cross-sectional z-score normalization and date-mean aggregation; 2020 subset is used for the standardized audit. | 1,330,165 params; 24 checkpoints, 122.7MB | 0.024410 | 0.037818 |

## Version Notes

`version1` is the smaller lite operator branch. It is useful because it keeps
the full factor-operator/MoE recipe while reducing the market feature set and
checkpoint footprint.

`version2` upgrades the lite branch to full market features. It is the best
pooled-IC compact neural branch in this retained large set.

`version3` keeps the review-stage ST-top-k idea. It is weaker on pooled IC than
the compact operator branches, but has competitive SN non-overlap IC and a
different normalization/aggregation path worth preserving for later ablation.

## Shared Files

Each version contains:

- `src/`: data, rolling, model, train, metrics, and visualization code.
- `configs/`: exact YAML config; `version3` also stores candidate configs.
- `weights/`: trained monthly checkpoints.
- `metrics/`: original outputs plus same-standard pooled/SN audit CSVs.
- `figures/`: original figures plus the generated dashboard.

No raw data, feature cache, or prediction parquet is included.
