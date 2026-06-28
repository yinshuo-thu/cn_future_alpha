# version1

Run: `factor_operator_extra_market_lite_seq_top48_huber_moe4_monthly12_val1`.

Lite factor-operator branch with extra market features, sequence top-48 operator gate, Huber loss, and a 4-expert MoE. This is the smaller large-model variant kept for architecture diversity and lower checkpoint footprint.

| Metric | Value |
| --- | ---: |
| Pooled IC | 0.027534 |
| SN non-overlap IC | 0.037190 |
| RankIC | 0.045261 |
| Params | 1,401,511 |
| Checkpoints | 12 monthly checkpoints, 2020M01-2020M12 |
