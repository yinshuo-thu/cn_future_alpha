# V7 Idea Validation Summary

- Train window: [2017-01-01, 2019-01-01)
- Eval window: [2019-01-01, 2020-01-01)
- Baseline: `transformer_v6_2019_validation_market_only_from_v4_nolf_e1`
- Decision gate: keep only when both Pooled IC and SN non-overlap IC improve versus the current retained model.

## Results

| experiment | Pooled_IC | SN_nonoverlap_IC | raw_nonoverlap_IC | dense_IC | merged_IC | SN_nonoverlap_RankIC | decision |
| --- | --- | --- | --- | --- | --- | --- | --- |
| baseline_v6_market_only | 0.062069 | 0.069863 | 0.078247 | 0.077234 | 0.061555 | 0.079789 | baseline |
| idea1_factor_k160_project | 0.059374 | 0.067541 | 0.074572 | 0.074801 | 0.058723 | 0.075977 | reject |
| idea1_factor_k96_scaled | 0.064172 | 0.070858 | 0.080524 | 0.080186 | 0.063401 | 0.080474 | keep |
| idea2_lowrank_replace | 0.054888 | 0.064736 | 0.070901 | 0.072032 | 0.054306 | 0.075681 | reject |
| idea2_lowrank_residual | 0.064778 | 0.069182 | 0.078453 | 0.078601 | 0.064243 | 0.078553 | reject: pooled up, SN down |
| idea3_meta_full | 0.055533 | 0.067231 | 0.075839 | 0.075686 | 0.055510 | 0.078886 | reject |
| idea3_meta_symbol_minute_scaled | 0.060745 | 0.070177 | 0.078329 | 0.078961 | 0.060532 | 0.079584 | reject |
| idea4_moe4_balance001 | 0.058668 | 0.067202 | 0.076829 | 0.075583 | 0.058150 | 0.078970 | reject |
| idea4_moe2_no_balance | 0.060294 | 0.064352 | 0.074821 | 0.075362 | 0.059287 | 0.075392 | reject |

## Retained Model

Retain `idea1_factor_k96_scaled`:

- Run directory: `/root/autodl-tmp/quant/end2end_30m/runs/transformer_v7_factor_bank_k96_scaled_project_e1`
- Report: `/root/autodl-tmp/quant/end2end_30m/reports/19_idea1_factor_bank_k96_scaled_project_e1.md`
- Prediction path: `/root/autodl-tmp/quant/end2end_30m/runs/transformer_v7_factor_bank_k96_scaled_project_e1/raw/predictions.parquet`

Architecture deltas versus v6 market-only:

- Enable `FactorOperatorBank`.
- Use windows `[3, 5, 8, 13, 21, 34, 55, 89]`.
- Build 483 online operators.
- Gate with sequence stats and keep `top_k=96`.
- Use `project` mode and `factor_scale_init=-2.0`.
- Do not retain LowRank input interaction, meta embeddings, or MoE head.

## Notes

- The first FactorOperatorBank attempt (`top_k=160`, no small residual scale) underperformed and had one skipped non-finite gradient batch.
- Scaling the factor bank down at initialization and reducing `top_k` fixed the instability and improved both decision metrics.
- LowRank residual was numerically stable and improved Pooled IC, but reduced SN non-overlap IC, so it was not retained.
- The second meta embedding attempt was stable and improved over full metadata, but still did not beat the retained FactorBank-only model.
- Both MoE attempts underperformed, including a simpler 2-expert no-balance version.
