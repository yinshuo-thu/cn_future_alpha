# V4-V6 Final Transformer End2End Report

- Final selected shape: `E2E_GatedMSPatch_MTL_DataLimited_v46` with TimeBiasAttention, SwiGLU+LayerScale, and market gating.
- Removed/rejected: layer fusion, RevIN, cross-section attention, cross-variate branch.
- Gate: 2019 validation requires both SN non-overlap IC and Pooled IC improvement.
- Final OOS: 2019 uses train [2017-01-01, 2019-01-01); 2020 uses train [2017-01-01, 2019-12-31).

## Selection Summary

| experiment | Pooled_IC | SN_nonoverlap_IC | raw_nonoverlap_IC | dense_IC | merged_IC | decision |
| --- | --- | --- | --- | --- | --- | --- |
| v1 baseline 2019 | 0.054609 | 0.064411 | 0.070722 | 0.070378 | 0.054452 | baseline |
| v4 full raw | 0.054555 | 0.065759 | 0.071702 | 0.070437 | 0.054245 | SN up, pooled flat/down; reject full v4 |
| v4 no layer fusion | 0.057418 | 0.068496 | 0.074273 | 0.071960 | 0.056907 | pass; keep TimeBias + SwiGLU/LayerScale, drop layer fusion |
| v5 full | 0.042990 | 0.061836 | 0.068106 | 0.063702 | 0.043440 | fail |
| v5 RevIN only | 0.048227 | 0.065293 | 0.069727 | 0.067010 | 0.048691 | fail; RevIN negative |
| v5 cross-section only | 0.053390 | 0.066734 | 0.075139 | 0.073693 | 0.053584 | fail vs v4 no layer fusion |
| v6 market + cross-variate | 0.055553 | 0.066867 | 0.074077 | 0.075049 | 0.055338 | fail vs v4; dense up but gate down |
| v6 market only | 0.062069 | 0.069863 | 0.078247 | 0.077234 | 0.061555 | pass; final 2019-selected shape |
| v6 cross-variate only | 0.053956 | 0.063309 | 0.069815 | 0.070851 | 0.053965 | fail; cross-variate negative |
| final minus TimeBias | 0.056728 | 0.065373 | 0.072314 | 0.072067 | 0.056364 | weaker; TimeBias useful |
| final minus SwiGLU+LayerScale | 0.057917 | 0.064135 | 0.074340 | 0.074995 | 0.058007 | weaker; SwiGLU+LayerScale useful |

## Final OOS Metrics

| period | pooled_IC | SN_nonoverlap_IC | raw_nonoverlap_IC | dense_IC | merged_IC | SN_nonoverlap_RankIC | n_scored |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 2019 | 0.062069 | 0.069863 | 0.078247 | 0.077234 | 0.061555 | 0.079789 | 2693337 |
| 2020 | 0.048159 | 0.061365 | 0.064287 | 0.070408 | 0.048316 | 0.067758 | 2827738 |
| 2019+2020_oos_walkforward | 0.053590 | 0.067774 | 0.075549 | 0.073734 | 0.053513 | 0.075749 | 5521075 |

## Improvement vs V1

| period | delta_pooled_IC | delta_SN_nonoverlap_IC | delta_raw_nonoverlap_IC | delta_dense_IC | delta_merged_IC |
| --- | --- | --- | --- | --- | --- |
| 2019 | 0.007460 | 0.005452 | 0.007525 | 0.006856 | 0.007103 |
| 2020 | 0.004581 | 0.002281 | 0.001458 | -0.000029 | 0.004834 |
| 2019+2020_oos_walkforward | 0.005712 | 0.003418 | 0.005670 | 0.003326 | 0.005763 |

## Artifacts

- Combined summary: `/root/autodl-tmp/quant/end2end_30m/runs/transformer_v6_market_only_2019_2020_oos_combined/summary.csv`
- Selection summary: `/root/autodl-tmp/quant/end2end_30m/runs/transformer_v6_market_only_2019_2020_oos_combined/selection_ablation_summary.csv`
- Final vs v1: `/root/autodl-tmp/quant/end2end_30m/runs/transformer_v6_market_only_2019_2020_oos_combined/final_vs_v1.csv`
- Final report: `/root/autodl-tmp/quant/end2end_30m/reports/18_v4_v6_final_market_only_oos.md`
