# Stage 1 Baseline Report

Experiment: `factor_operator_extra_market_full_seq_top48_huber_moe4_monthly12_val1`

## Architecture

Raw 1-minute OHLCV/amount/OI -> strictly causal rolling z-score -> low-rank FM-style interaction block -> PatchTST-style patch embedding -> pre-norm Transformer -> learnable multi-layer output aggregation -> attention/mean/CLS pooling -> MLP prediction head.

## Key Metrics

- Merged test IC: `0.032294`
- Merged test RankIC: `0.050446`
- Monthly ICIR: `1.862610`
- Scored rows: `2939535`

## Rolling Splits

| split | train_start | train_end | test_start | test_end | train_windows | val_windows | test_windows | ic | rank_ic |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2020M01 | 2018-01-01 | 2020-01-01 | 2020-01-01 | 2020-02-01 | 220000 | 40000 | 206469 | 0.050950 | 0.068150 |
| 2020M02 | 2018-01-01 | 2020-02-01 | 2020-02-01 | 2020-03-01 | 220000 | 40000 | 163452 | 0.020312 | 0.062707 |
| 2020M03 | 2018-01-01 | 2020-03-01 | 2020-03-01 | 2020-04-01 | 220000 | 40000 | 180119 | 0.028505 | 0.042970 |
| 2020M04 | 2018-01-01 | 2020-04-01 | 2020-04-01 | 2020-05-01 | 220000 | 40000 | 175830 | 0.045546 | 0.073054 |
| 2020M05 | 2018-01-01 | 2020-05-01 | 2020-05-01 | 2020-06-01 | 220000 | 40000 | 245850 | 0.066868 | 0.081601 |
| 2020M06 | 2018-01-01 | 2020-06-01 | 2020-06-01 | 2020-07-01 | 220000 | 40000 | 269027 | 0.039276 | 0.060792 |
| 2020M07 | 2018-01-01 | 2020-07-01 | 2020-07-01 | 2020-08-01 | 220000 | 40000 | 316667 | 0.029672 | 0.047299 |
| 2020M08 | 2018-01-01 | 2020-08-01 | 2020-08-01 | 2020-09-01 | 220000 | 40000 | 286473 | 0.039246 | 0.045859 |
| 2020M09 | 2018-01-01 | 2020-09-01 | 2020-09-01 | 2020-10-01 | 220000 | 40000 | 296947 | 0.052155 | 0.061047 |
| 2020M10 | 2018-01-01 | 2020-10-01 | 2020-10-01 | 2020-11-01 | 220000 | 40000 | 219635 | 0.006948 | 0.028616 |
| 2020M11 | 2018-01-01 | 2020-11-01 | 2020-11-01 | 2020-12-01 | 220000 | 40000 | 287578 | 0.014784 | 0.033344 |
| 2020M12 | 2018-01-01 | 2020-12-01 | 2020-12-01 | 2021-01-01 | 220000 | 40000 | 291488 | 0.013203 | 0.037134 |

## Artifacts

- split_metrics: `/root/autodl-tmp/quant/end2end/runs/factor_operator_extra_market_full_seq_top48_huber_moe4_monthly12_val1/split_metrics.csv`
- yearly_metrics: `/root/autodl-tmp/quant/end2end/runs/factor_operator_extra_market_full_seq_top48_huber_moe4_monthly12_val1/yearly_metrics.csv`
- monthly_metrics: `/root/autodl-tmp/quant/end2end/runs/factor_operator_extra_market_full_seq_top48_huber_moe4_monthly12_val1/monthly_metrics.csv`
- symbol_metrics: `/root/autodl-tmp/quant/end2end/runs/factor_operator_extra_market_full_seq_top48_huber_moe4_monthly12_val1/symbol_metrics.csv`
- metrics: `/root/autodl-tmp/quant/end2end/runs/factor_operator_extra_market_full_seq_top48_huber_moe4_monthly12_val1/metrics.json`
- figure: `/root/autodl-tmp/quant/end2end/figures/factor_operator_extra_market_full_seq_top48_huber_moe4_monthly12_val1/factor_operator_extra_market_full_seq_top48_huber_moe4_monthly12_val1_pred_distribution.png`
- figure: `/root/autodl-tmp/quant/end2end/figures/factor_operator_extra_market_full_seq_top48_huber_moe4_monthly12_val1/factor_operator_extra_market_full_seq_top48_huber_moe4_monthly12_val1_label_distribution.png`
- figure: `/root/autodl-tmp/quant/end2end/figures/factor_operator_extra_market_full_seq_top48_huber_moe4_monthly12_val1/factor_operator_extra_market_full_seq_top48_huber_moe4_monthly12_val1_binned_pred_label.png`
- figure: `/root/autodl-tmp/quant/end2end/figures/factor_operator_extra_market_full_seq_top48_huber_moe4_monthly12_val1/factor_operator_extra_market_full_seq_top48_huber_moe4_monthly12_val1_monthly_ic.png`
- figure: `/root/autodl-tmp/quant/end2end/figures/factor_operator_extra_market_full_seq_top48_huber_moe4_monthly12_val1/factor_operator_extra_market_full_seq_top48_huber_moe4_monthly12_val1_cumulative_ic.png`

## Notes

- Rolling normalization is history-only via shifted rolling windows within symbol/session groups.
- The baseline is intentionally compact for first-pass reliability; larger widths, more windows, and additional ideas should be tested through the ablation harness next.
