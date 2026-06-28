# Data Notes

Raw data is not stored in this repository. Download the 1-minute China futures
dataset from Kaggle:

https://www.kaggle.com/datasets/wentinglu/highfrequency-futures-data-china/data

The original experiments used local paths under `/root/autodl-tmp/quant/data/raw`
and `/root/autodl-tmp/fu-alpha-research`. On a new machine, either recreate
those paths or update the path constants/YAML configs in the copied scripts.

Evaluation excludes financial index/bond symbols listed in the configs:
`T`, `TF`, `TS`, `IF`, `IC`, `IH`.
