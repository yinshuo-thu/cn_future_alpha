#!/usr/bin/env python3
"""2019-only top-K sweep for FU new-factor Ridge/LGB/MLP components.

This script consumes the existing 2019Q4 leave-one/shuffle selection scores
and retrains component models with top-K feature subsets.  It only evaluates
2019Q4 validation predictions, so it is safe to use for choosing K before the
final 2020 test run.
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "/root/autodl-tmp/quant/ML")

from fu_newfactor_three_model_ensemble import (  # noqa: E402
    DEFAULT_EXPR,
    DEFAULT_FEATURES,
    DEFAULT_OUT,
    DEFAULT_SAMPLE_DIR,
    MLPTrainConfig,
    fit_ic_weights_from_stats,
    fit_lgb,
    fit_mlp_model,
    fit_ridge,
    load_samples,
    month_range,
    predict_mlp,
    predict_ridge,
    save_pred,
    scrub_matrix,
    val_arrays,
    x_y,
)
from rolling_factor_model_eval import compute_ic  # noqa: E402


BASE_DIR = DEFAULT_OUT
OUT_DIR = Path("/root/autodl-tmp/quant/ML/strict_opt_results/fu_newfactor_topk_sweep")


def write_json(path: Path, obj: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def top_features(kind: str, k: int, base_dir: Path) -> list[str]:
    files = {
        "ridge": "selection_ridge_leave_one_2019q4.csv",
        "lgb": "selection_lgb_shuffle_2019q4.csv",
        "mlp": "selection_mlp_shuffle_2019q4.csv",
    }
    path = base_dir / files[kind]
    df = pd.read_csv(path).sort_values("delta_ic", ascending=False)
    if k <= 0:
        df = df[df["delta_ic"] > 0]
    else:
        df = df.head(k)
    return df["factor"].astype(str).tolist()


def component_path(out_dir: Path, kind: str, k: int) -> Path:
    return out_dir / "validation_parts" / f"{kind}_k{k}.parquet"


def fit_component(kind: str, k: int, features: list[str], args: argparse.Namespace) -> dict[str, object]:
    out_path = component_path(args.out_dir, kind, k)
    if out_path.exists() and not args.force:
        pred = pd.read_parquet(out_path)
        return {
            "component": f"{kind}_k{k}",
            "kind": kind,
            "k": k,
            "features": len(features),
            "val_pred_ic": compute_ic(pred["pred"].to_numpy(), pred["label"].to_numpy()),
            "val_pred_xsz_ic": compute_ic(pred["pred_xsz"].to_numpy(), pred["label"].to_numpy()),
            "path": str(out_path),
            "cached": True,
        }

    train = load_samples(args.sample_dir, month_range("2018-01", "2019-09"), features)
    val = load_samples(args.sample_dir, month_range("2019-10", "2019-12"), features)
    print(f"[topk][{kind}_k{k}] train={len(train)} val={len(val)} features={len(features)}", flush=True)

    if kind == "ridge":
        x, y = x_y(train, features)
        model = fit_ridge(x, y, args.ridge_alpha)
        xv, _label, _bounds, meta = val_arrays(val, features)
        pred = predict_ridge(model, xv)
        del x, y, xv, model
    elif kind == "lgb":
        x, y = x_y(train, features, args.lgb_target_col)
        model = fit_lgb(x, y, args)
        xv, _label, _bounds, meta = val_arrays(val, features)
        pred = model.booster_.predict(xv, num_threads=args.threads)
        del x, y, xv, model
    elif kind == "mlp":
        cfg = MLPTrainConfig(
            hidden=args.mlp_hidden,
            dropout=args.mlp_dropout,
            epochs=args.mlp_epochs,
            batch_size=args.mlp_batch_size,
            seed=args.seed + k,
        )
        model, mean, scale, device = fit_mlp_model(
            train[["datetime", "label", "label_xsz"] + features],
            features,
            "2019-10",
            cfg,
        )
        xv, _label, _bounds, meta = val_arrays(val, features)
        pred = predict_mlp(model, scrub_matrix(xv), mean, scale, device, args.mlp_predict_chunk)
        del xv, model, mean, scale
        if str(device) == "cuda":
            import torch

            torch.cuda.empty_cache()
    else:
        raise ValueError(kind)

    save_pred(meta, pred, out_path)
    saved = pd.read_parquet(out_path)
    del train, val, pred, saved
    gc.collect()
    saved = pd.read_parquet(out_path)
    return {
        "component": f"{kind}_k{k}",
        "kind": kind,
        "k": k,
        "features": len(features),
        "val_pred_ic": compute_ic(saved["pred"].to_numpy(), saved["label"].to_numpy()),
        "val_pred_xsz_ic": compute_ic(saved["pred_xsz"].to_numpy(), saved["label"].to_numpy()),
        "path": str(out_path),
        "cached": False,
    }


def load_component(out_dir: Path, name: str) -> pd.DataFrame:
    path = out_dir / "validation_parts" / f"{name}.parquet"
    cur = pd.read_parquet(path, columns=["symbol", "datetime", "label", "pred_xsz"])
    cur["datetime"] = pd.to_datetime(cur["datetime"])
    return cur.rename(columns={"pred_xsz": name})


def combo_summary(out_dir: Path, components: list[str]) -> pd.DataFrame:
    by_kind: dict[str, list[str]] = {"ridge": [], "lgb": [], "mlp": []}
    for name in components:
        kind = name.split("_k", 1)[0]
        by_kind[kind].append(name)

    rows = []
    for ridge in by_kind["ridge"]:
        for lgb in by_kind["lgb"]:
            for mlp in by_kind["mlp"]:
                names = [ridge, lgb, mlp]
                base = None
                for name in names:
                    cur = load_component(out_dir, name)
                    if base is None:
                        base = cur
                    else:
                        base = base.merge(cur[["symbol", "datetime", name]], on=["symbol", "datetime"], how="inner")
                assert base is not None
                x = base[names].to_numpy(np.float64)
                y = base["label"].to_numpy(np.float64)
                mask = np.isfinite(y) & np.all(np.isfinite(x), axis=1)
                x = x[mask]
                y = y[mask]
                c = x.T @ y
                g = x.T @ x
                yy = float(y @ y)
                w, fit_ic = fit_ic_weights_from_stats(c, g, yy, np.zeros(3), np.ones(3))
                pred = base[names].to_numpy(np.float32) @ w.astype(np.float32)
                rows.append(
                    {
                        "ridge": ridge,
                        "lgb": lgb,
                        "mlp": mlp,
                        "val_fit_ic": float(fit_ic),
                        "val_pred_ic": compute_ic(pred, base["label"].to_numpy()),
                        "w_ridge": float(w[0]),
                        "w_lgb": float(w[1]),
                        "w_mlp": float(w[2]),
                    }
                )
    return pd.DataFrame(rows).sort_values("val_fit_ic", ascending=False).reset_index(drop=True)


def parse_csv_ints(value: str) -> list[int]:
    return [int(x.strip()) for x in value.split(",") if x.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-dir", type=Path, default=BASE_DIR)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--feature-file", type=Path, default=DEFAULT_FEATURES)
    parser.add_argument("--expression-file", type=Path, default=DEFAULT_EXPR)
    parser.add_argument("--sample-dir", type=Path, default=DEFAULT_SAMPLE_DIR)
    parser.add_argument("--ridge-ks", type=parse_csv_ints, default=parse_csv_ints("128,256,512,885"))
    parser.add_argument("--lgb-ks", type=parse_csv_ints, default=parse_csv_ints("64,128,256,512,838"))
    parser.add_argument("--mlp-ks", type=parse_csv_ints, default=parse_csv_ints("64,128,256,537"))
    parser.add_argument("--seed", type=int, default=20260626)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--ridge-alpha", type=float, default=1.0)
    parser.add_argument("--lgb-estimators", type=int, default=260)
    parser.add_argument("--lgb-lr", type=float, default=0.04)
    parser.add_argument("--lgb-leaves", type=int, default=63)
    parser.add_argument("--lgb-min-child", type=int, default=120)
    parser.add_argument("--lgb-lambda", type=float, default=4.0)
    parser.add_argument("--lgb-colsample", type=float, default=0.65)
    parser.add_argument("--lgb-target-col", choices=["label", "label_xsz"], default="label_xsz")
    parser.add_argument("--mlp-hidden", type=int, default=192)
    parser.add_argument("--mlp-dropout", type=float, default=0.12)
    parser.add_argument("--mlp-epochs", type=int, default=3)
    parser.add_argument("--mlp-batch-size", type=int, default=8192)
    parser.add_argument("--mlp-predict-chunk", type=int, default=131072)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    names = []
    for kind, ks in [("ridge", args.ridge_ks), ("lgb", args.lgb_ks), ("mlp", args.mlp_ks)]:
        for k in ks:
            features = top_features(kind, k, args.base_dir)
            row = fit_component(kind, k, features, args)
            rows.append(row)
            names.append(row["component"])
            pd.DataFrame(rows).to_csv(args.out_dir / "component_validation_summary.csv", index=False)
    combos = combo_summary(args.out_dir, names)
    combos.to_csv(args.out_dir / "combo_validation_summary.csv", index=False)
    best = combos.iloc[0].to_dict()
    write_json(
        args.out_dir / "sweep_metadata.json",
        {
            "selection_scores_source": str(args.base_dir),
            "validation_window": "2019Q4",
            "train_window": "2018-01..2019-09 samples",
            "uses_2020_for_selection": False,
            "best_combo": best,
        },
    )
    print("[topk] component summary", flush=True)
    print(pd.DataFrame(rows).sort_values("val_pred_xsz_ic", ascending=False).to_string(index=False), flush=True)
    print("[topk] best combos", flush=True)
    print(combos.head(20).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
