# ML Single Models

This folder keeps the three useful strict single ML models and the
infrastructure needed to rebuild them from raw futures bars.

## Structure

| Path | Purpose |
| --- | --- |
| `model/` | Factor-panel builders, rolling evaluators, MLP/LGB/Ridge implementations, and model-specific optimization scripts. |
| `scripts/` | Reproduction and audit entry points, including `run_single_model.py`. |
| `configs/` | Selected feature lists and 2019-only selected configs. |
| `weights/` | Lightweight retained postprocess states. Rolling tree/neural/ridge base models are retrained from raw factors. |
| `metrics/` | Pooled IC, SN non-overlap IC, monthly IC, 20-bin return, and selection audit CSVs. |
| `figures/` | Dashboard PNGs for the three single models. |

## Model Table

| Model | Core Architecture / Idea | Main Optimization | Pooled IC | SN non-overlap IC |
| --- | --- | --- | ---: | ---: |
| `mlp_time120_slope_a025_strong` | Rolling MLP over effective factor features; postprocess blends raw/center/z views and applies time-bucket slope calibration. | 2019-only calibration screen; time120 strong-slope multiplier; no 2020 labels used for selection. | 0.050756 | 0.065097 |
| `lgb_ref_time90_a1_signed_abs12_a08` | Strict rolling LightGBM stream with cross-sectional views and signed-absolute shape calibration. | 2019 shape/refinement search, then recent-weak selector with `time90`, `a1`, signed abs bucket strength `0.8`. | 0.050034 | 0.065138 |
| `ridge_simplex_basic_full2019` | Rolling Ridge stream plus small view ensemble over raw/xcenter/xsz/xrank-like transforms. | Internal 2019 selection and nonnegative simplex weights fit before the 2020 audit. | 0.042481 | 0.064183 |

## What Was Worth Keeping

- Rolling train-before-test infrastructure and label embargo handling.
- Cross-sectional prediction views (`xcenter`, `xsz`, `xrank`) that align better
  with IC evaluation than raw predictions alone.
- 2019-only calibration/search code for MLP and LGB, which improves stability
  without touching 2020 labels.
- Ridge postprocess weights, because they are compact, interpretable, and easy
  to audit.

## Commands

Run a model implementation by name and pass through the original script options:

```bash
cd /root/jump_model/ML_single
python scripts/run_single_model.py --model mlp -- --help
python scripts/run_single_model.py --model lgb -- --help
python scripts/run_single_model.py --model ridge -- --help
```

The raw data path is configured in the copied infrastructure. On a new machine,
download the Kaggle raw data and update the config/path constants as needed,
then rebuild factor panels before training. Regenerate dashboards from the
project root with `python tools/generate_ml_audit_assets.py`.

## Dashboards

- `figures/mlp_time120_slope_a025_strong_dashboard.png`
- `figures/lgb_ref_time90_a1_signed_abs12_a08_dashboard.png`
- `figures/ridge_simplex_basic_full2019_dashboard.png`
