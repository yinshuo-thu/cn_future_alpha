# Stage 1 Baseline Report

Experiment: `factor_operator_extra_market_lite_seq_top48_huber_moe4_monthly12_val1`

## Architecture

Raw 1-minute OHLCV/amount/OI -> strictly causal rolling z-score -> low-rank FM-style interaction block -> PatchTST-style patch embedding -> pre-norm Transformer -> learnable multi-layer output aggregation -> attention/mean/CLS pooling -> MLP prediction head.

## Key Metrics

- Merged test IC: `0.027534`
- Merged test RankIC: `0.045261`
- Monthly ICIR: `0.887050`
- Scored rows: `2939535`

## Rolling Splits

| split | train_start | train_end | test_start | test_end | train_windows | val_windows | test_windows | ic | rank_ic |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2020M01 | 2018-01-01 | 2020-01-01 | 2020-01-01 | 2020-02-01 | 220000 | 40000 | 206469 | 0.032179 | 0.057191 |
| 2020M02 | 2018-01-01 | 2020-02-01 | 2020-02-01 | 2020-03-01 | 220000 | 40000 | 163452 | 0.001403 | 0.054139 |
| 2020M03 | 2018-01-01 | 2020-03-01 | 2020-03-01 | 2020-04-01 | 220000 | 40000 | 180119 | 0.025987 | 0.040917 |
| 2020M04 | 2018-01-01 | 2020-04-01 | 2020-04-01 | 2020-05-01 | 220000 | 40000 | 175830 | 0.000356 | 0.041456 |
| 2020M05 | 2018-01-01 | 2020-05-01 | 2020-05-01 | 2020-06-01 | 220000 | 40000 | 245850 | 0.075792 | 0.087352 |
| 2020M06 | 2018-01-01 | 2020-06-01 | 2020-06-01 | 2020-07-01 | 220000 | 40000 | 269027 | 0.040142 | 0.061741 |
| 2020M07 | 2018-01-01 | 2020-07-01 | 2020-07-01 | 2020-08-01 | 220000 | 40000 | 316667 | 0.029901 | 0.042931 |
| 2020M08 | 2018-01-01 | 2020-08-01 | 2020-08-01 | 2020-09-01 | 220000 | 40000 | 286473 | 0.028297 | 0.034365 |
| 2020M09 | 2018-01-01 | 2020-09-01 | 2020-09-01 | 2020-10-01 | 220000 | 40000 | 296947 | 0.059942 | 0.068296 |
| 2020M10 | 2018-01-01 | 2020-10-01 | 2020-10-01 | 2020-11-01 | 220000 | 40000 | 219635 | -0.032936 | -0.011771 |
| 2020M11 | 2018-01-01 | 2020-11-01 | 2020-11-01 | 2020-12-01 | 220000 | 40000 | 287578 | 0.012921 | 0.032924 |
| 2020M12 | 2018-01-01 | 2020-12-01 | 2020-12-01 | 2021-01-01 | 220000 | 40000 | 291488 | 0.026877 | 0.049603 |

## Artifacts

- split_metrics: `/root/autodl-tmp/quant/end2end/runs/factor_operator_extra_market_lite_seq_top48_huber_moe4_monthly12_val1/split_metrics.csv`
- yearly_metrics: `/root/autodl-tmp/quant/end2end/runs/factor_operator_extra_market_lite_seq_top48_huber_moe4_monthly12_val1/yearly_metrics.csv`
- monthly_metrics: `/root/autodl-tmp/quant/end2end/runs/factor_operator_extra_market_lite_seq_top48_huber_moe4_monthly12_val1/monthly_metrics.csv`
- symbol_metrics: `/root/autodl-tmp/quant/end2end/runs/factor_operator_extra_market_lite_seq_top48_huber_moe4_monthly12_val1/symbol_metrics.csv`
- metrics: `/root/autodl-tmp/quant/end2end/runs/factor_operator_extra_market_lite_seq_top48_huber_moe4_monthly12_val1/metrics.json`
- figure: `/root/autodl-tmp/quant/end2end/figures/factor_operator_extra_market_lite_seq_top48_huber_moe4_monthly12_val1/factor_operator_extra_market_lite_seq_top48_huber_moe4_monthly12_val1_pred_distribution.png`
- figure: `/root/autodl-tmp/quant/end2end/figures/factor_operator_extra_market_lite_seq_top48_huber_moe4_monthly12_val1/factor_operator_extra_market_lite_seq_top48_huber_moe4_monthly12_val1_label_distribution.png`
- figure: `/root/autodl-tmp/quant/end2end/figures/factor_operator_extra_market_lite_seq_top48_huber_moe4_monthly12_val1/factor_operator_extra_market_lite_seq_top48_huber_moe4_monthly12_val1_binned_pred_label.png`
- figure: `/root/autodl-tmp/quant/end2end/figures/factor_operator_extra_market_lite_seq_top48_huber_moe4_monthly12_val1/factor_operator_extra_market_lite_seq_top48_huber_moe4_monthly12_val1_monthly_ic.png`
- figure: `/root/autodl-tmp/quant/end2end/figures/factor_operator_extra_market_lite_seq_top48_huber_moe4_monthly12_val1/factor_operator_extra_market_lite_seq_top48_huber_moe4_monthly12_val1_cumulative_ic.png`

## Notes

- Rolling normalization is history-only via shifted rolling windows within symbol/session groups.
- The baseline is intentionally compact for first-pass reliability; larger widths, more windows, and additional ideas should be tested through the ablation harness next.
