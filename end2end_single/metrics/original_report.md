# Stage 1 Baseline Report

Experiment: `factor_operator_market_full_oldarch_fullsample_m05_m12_raw_seed44`

## Architecture

Raw 1-minute OHLCV/amount/OI -> strictly causal rolling z-score -> low-rank FM-style interaction block -> PatchTST-style patch embedding -> pre-norm Transformer -> learnable multi-layer output aggregation -> attention/mean/CLS pooling -> MLP prediction head.

## Key Metrics

- Merged test IC: `0.039660`
- Merged test RankIC: `0.049840`
- Monthly ICIR: `2.280238`
- Scored rows: `2213665`

## Rolling Splits

| split | train_start | train_end | test_start | test_end | train_windows | val_windows | test_windows | ic | rank_ic |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2020M05 | 2018-01-01 | 2020-05-01 | 2020-05-01 | 2020-06-01 | 6114728 | 175830 | 245850 | 0.072084 | 0.092711 |
| 2020M06 | 2018-01-01 | 2020-06-01 | 2020-06-01 | 2020-07-01 | 6290558 | 245850 | 269027 | 0.031290 | 0.049958 |
| 2020M07 | 2018-01-01 | 2020-07-01 | 2020-07-01 | 2020-08-01 | 6536408 | 269027 | 316667 | 0.058381 | 0.062796 |
| 2020M08 | 2018-01-01 | 2020-08-01 | 2020-08-01 | 2020-09-01 | 6805435 | 316667 | 286473 | 0.047720 | 0.062036 |
| 2020M09 | 2018-01-01 | 2020-09-01 | 2020-09-01 | 2020-10-01 | 7122102 | 286473 | 296947 | 0.028112 | 0.032189 |
| 2020M10 | 2018-01-01 | 2020-10-01 | 2020-10-01 | 2020-11-01 | 7408575 | 296947 | 219635 | 0.023750 | 0.029693 |
| 2020M11 | 2018-01-01 | 2020-11-01 | 2020-11-01 | 2020-12-01 | 7705522 | 219635 | 287578 | 0.021466 | 0.036151 |
| 2020M12 | 2018-01-01 | 2020-12-01 | 2020-12-01 | 2021-01-01 | 7925157 | 287578 | 291488 | 0.049472 | 0.035654 |

## Artifacts

- split_metrics: `/root/autodl-tmp/quant/end2end/runs/factor_operator_market_full_oldarch_fullsample_m05_m12_raw_seed44/split_metrics.csv`
- yearly_metrics: `/root/autodl-tmp/quant/end2end/runs/factor_operator_market_full_oldarch_fullsample_m05_m12_raw_seed44/yearly_metrics.csv`
- monthly_metrics: `/root/autodl-tmp/quant/end2end/runs/factor_operator_market_full_oldarch_fullsample_m05_m12_raw_seed44/monthly_metrics.csv`
- symbol_metrics: `/root/autodl-tmp/quant/end2end/runs/factor_operator_market_full_oldarch_fullsample_m05_m12_raw_seed44/symbol_metrics.csv`
- metrics: `/root/autodl-tmp/quant/end2end/runs/factor_operator_market_full_oldarch_fullsample_m05_m12_raw_seed44/metrics.json`
- figure: `/root/autodl-tmp/quant/end2end/figures/factor_operator_market_full_oldarch_fullsample_m05_m12_raw_seed44/factor_operator_market_full_oldarch_fullsample_m05_m12_raw_seed44_pred_distribution.png`
- figure: `/root/autodl-tmp/quant/end2end/figures/factor_operator_market_full_oldarch_fullsample_m05_m12_raw_seed44/factor_operator_market_full_oldarch_fullsample_m05_m12_raw_seed44_label_distribution.png`
- figure: `/root/autodl-tmp/quant/end2end/figures/factor_operator_market_full_oldarch_fullsample_m05_m12_raw_seed44/factor_operator_market_full_oldarch_fullsample_m05_m12_raw_seed44_binned_pred_label.png`
- figure: `/root/autodl-tmp/quant/end2end/figures/factor_operator_market_full_oldarch_fullsample_m05_m12_raw_seed44/factor_operator_market_full_oldarch_fullsample_m05_m12_raw_seed44_monthly_ic.png`
- figure: `/root/autodl-tmp/quant/end2end/figures/factor_operator_market_full_oldarch_fullsample_m05_m12_raw_seed44/factor_operator_market_full_oldarch_fullsample_m05_m12_raw_seed44_cumulative_ic.png`

## Notes

- Rolling normalization is history-only via shifted rolling windows within symbol/session groups.
- The baseline is intentionally compact for first-pass reliability; larger widths, more windows, and additional ideas should be tested through the ablation harness next.
