#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import json
import math
import os
import sys
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "/root/autodl-tmp/fu-alpha-research/src")
sys.path.insert(0, "/root/autodl-tmp/quant/ML")

from fu_alpha_research.config import load_config
from fu_alpha_research.feature_matrix import FeatureMatrix, read_feature_list
from fu_alpha_research.mlp import make_mlp, recency_weights, scrub_matrix, weighted_loss
from rolling_factor_model_eval import add_cross_sectional_norms, compute_ic, fit_ic_weights_from_stats, period_ic


warnings.filterwarnings("ignore", message="X does not have valid feature names")


ROOT = Path("/root/autodl-tmp/fu-alpha-research")
DEFAULT_FEATURES = ROOT / "outputs/model_feature_sets/new1000_model_union.txt"
DEFAULT_EXPR = ROOT / "outputs/expression_sets/combined_for_new1000_models.csv"
DEFAULT_SAMPLE_DIR = ROOT / "outputs/mlp_samples_new1000/is"
DEFAULT_OUT = Path("/root/autodl-tmp/quant/ML/strict_opt_results/fu_newfactor_three_model")


@dataclass(frozen=True)
class MLPTrainConfig:
    hidden: int = 192
    dropout: float = 0.12
    epochs: int = 5
    batch_size: int = 8192
    lr: float = 1e-3
    weight_decay: float = 1e-4
    half_life_months: float = 12.0
    seed: int = 20260624


def month_range(start: str, end: str) -> list[str]:
    return [str(x) for x in pd.period_range(start, end, freq="M")]


def prev_month(month: str) -> str:
    return str(pd.Period(month, freq="M") - 1)


def write_json(path: Path, obj: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_list(path: Path, values: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(values) + "\n", encoding="utf-8")


def load_samples(sample_dir: Path, months: list[str], features: list[str]) -> pd.DataFrame:
    cols = ["symbol", "datetime", "label", "label_xsz"] + features
    parts = []
    cfg = None
    matrix = None
    for month in months:
        path = sample_dir / f"{month}.parquet"
        try:
            part = pd.read_parquet(path, columns=cols)
        except Exception as exc:
            if "No match for FieldRef" not in str(exc):
                raise
            if cfg is None:
                cfg = load_config(str(ROOT / "configs/futures.yaml"))
                matrix = FeatureMatrix(cfg, DEFAULT_EXPR)
            assert matrix is not None
            full = matrix.read_month(month, features).dropna(subset=["label"]).copy()
            g = full.groupby("datetime", sort=False)["label"]
            full["label_xsz"] = ((full["label"] - g.transform("mean")) / (g.transform("std") + 1e-9)).clip(-8, 8).astype(np.float32)
            sample_n = 30_000
            if len(full) > sample_n:
                rng = np.random.default_rng(20260625 + int(month.replace("-", "")))
                idx = np.sort(rng.choice(len(full), sample_n, replace=False))
                full = full.iloc[idx].copy()
            part = full[cols]
            print(f"[load-sample] {month} fallback_full rows={len(part)}", flush=True)
        else:
            print(f"[load-sample] {month} rows={len(part)}", flush=True)
        part["datetime"] = pd.to_datetime(part["datetime"])
        parts.append(part)
    out = pd.concat(parts, ignore_index=True)
    return out


def x_y(df: pd.DataFrame, features: list[str], target: str = "label_xsz") -> tuple[np.ndarray, np.ndarray]:
    y = df[target].to_numpy(np.float32, copy=False)
    mask = np.isfinite(y)
    x = scrub_matrix(df.loc[mask, features].to_numpy(np.float32, copy=False))
    y = np.clip(y[mask], -8, 8).astype(np.float32)
    return x, y


def standardize_fit(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = x.mean(axis=0, dtype=np.float64).astype(np.float32)
    scale = np.maximum(x.std(axis=0, dtype=np.float64), 1e-6).astype(np.float32)
    return mean, scale


def group_bounds(datetimes: pd.Series) -> list[tuple[int, int]]:
    values = pd.to_datetime(datetimes).to_numpy()
    if len(values) == 0:
        return []
    change = np.flatnonzero(values[1:] != values[:-1]) + 1
    edges = np.r_[0, change, len(values)]
    return [(int(edges[i]), int(edges[i + 1])) for i in range(len(edges) - 1)]


def pred_xsz_ic(pred: np.ndarray, label: np.ndarray, bounds: list[tuple[int, int]]) -> float:
    xy = xx = yy = 0.0
    for start, end in bounds:
        p = pred[start:end].astype(np.float64, copy=False)
        y = label[start:end].astype(np.float64, copy=False)
        if len(p) < 2:
            continue
        sd = float(np.nanstd(p, ddof=1))
        if not np.isfinite(sd) or sd <= 1e-12:
            continue
        z = (p - float(np.nanmean(p))) / (sd + 1e-9)
        mask = np.isfinite(y)
        if not mask.any():
            continue
        zv = z[mask]
        yv = y[mask]
        xy += float(np.dot(zv, yv))
        xx += float(np.dot(zv, zv))
        yy += float(np.dot(yv, yv))
    return xy / math.sqrt(max(xx * yy, 1e-30))


def matrix_pred_xsz_ic(preds: np.ndarray, label: np.ndarray, bounds: list[tuple[int, int]]) -> np.ndarray:
    preds = np.asarray(preds, dtype=np.float64)
    out_len = preds.shape[1]
    xy = np.zeros(out_len, dtype=np.float64)
    xx = np.zeros(out_len, dtype=np.float64)
    yy = 0.0
    for start, end in bounds:
        p = preds[start:end]
        y = label[start:end].astype(np.float64, copy=False)
        if len(p) < 2:
            continue
        mu = np.nanmean(p, axis=0)
        sd = np.nanstd(p, axis=0, ddof=1)
        ok = np.isfinite(sd) & (sd > 1e-12)
        if not ok.any():
            continue
        z = (p[:, ok] - mu[ok]) / (sd[ok] + 1e-9)
        mask = np.isfinite(y)
        if not mask.any():
            continue
        zv = z[mask]
        yv = y[mask]
        xy[ok] += zv.T @ yv
        xx[ok] += np.sum(zv * zv, axis=0)
        yy += float(np.dot(yv, yv))
    return xy / np.sqrt(np.maximum(xx * yy, 1e-30))


def val_arrays(val: pd.DataFrame, features: list[str]) -> tuple[np.ndarray, np.ndarray, list[tuple[int, int]], pd.DataFrame]:
    frame = val.sort_values(["datetime", "symbol"]).reset_index(drop=True)
    label = frame["label"].to_numpy(np.float64, copy=False)
    bounds = group_bounds(frame["datetime"])
    meta = frame[["symbol", "datetime", "label"]].copy()
    x = scrub_matrix(frame[features].to_numpy(np.float32, copy=False))
    return x, label, bounds, meta


def fit_ridge(x: np.ndarray, y: np.ndarray, alpha: float) -> dict[str, np.ndarray | float]:
    mean, scale = standardize_fit(x)
    xz = ((x - mean) / scale).astype(np.float32)
    y_mean = float(y.mean())
    y0 = y.astype(np.float64) - y_mean
    gram = (xz.T @ xz).astype(np.float64) / max(len(xz), 1)
    cov = (xz.T @ y0).astype(np.float64) / max(len(xz), 1)
    weight = np.linalg.solve(gram + alpha * np.eye(xz.shape[1], dtype=np.float64), cov).astype(np.float32)
    return {"mean": mean, "scale": scale, "weight": weight, "y_mean": y_mean, "alpha": float(alpha)}


def predict_ridge(model: dict[str, np.ndarray | float], x: np.ndarray) -> np.ndarray:
    mean = model["mean"]
    scale = model["scale"]
    weight = model["weight"]
    y_mean = float(model["y_mean"])
    assert isinstance(mean, np.ndarray) and isinstance(scale, np.ndarray) and isinstance(weight, np.ndarray)
    return (((x - mean) / scale).astype(np.float32) @ weight + y_mean).astype(np.float32)


def ridge_leave_one_select(
    train: pd.DataFrame,
    val: pd.DataFrame,
    features: list[str],
    out_dir: Path,
    alpha: float,
    block_size: int,
) -> list[str]:
    path = out_dir / "selection_ridge_leave_one_2019q4.csv"
    if path.exists():
        result = pd.read_csv(path)
        return result[result["retained"]]["factor"].astype(str).tolist()

    x, y = x_y(train, features)
    mean, scale = standardize_fit(x)
    xz = ((x - mean) / scale).astype(np.float32)
    y_mean = float(y.mean())
    y0 = y.astype(np.float64) - y_mean
    gram = (xz.T @ xz).astype(np.float64) / max(len(xz), 1)
    cov = (xz.T @ y0).astype(np.float64) / max(len(xz), 1)
    k_mat = gram + alpha * np.eye(len(features), dtype=np.float64)
    k_inv = np.linalg.inv(k_mat)
    weight = k_inv @ cov
    diag = np.diag(k_inv)
    del x, xz, gram, cov, k_mat
    gc.collect()

    xv, label, bounds, _meta = val_arrays(val, features)
    xvz = ((xv - mean) / scale).astype(np.float32)
    del xv
    full_pred = xvz @ weight + y_mean
    base_ic = pred_xsz_ic(full_pred, label, bounds)
    rows = []
    for start in range(0, len(features), block_size):
        end = min(start + block_size, len(features))
        idx = np.arange(start, end)
        h = xvz @ k_inv[:, idx]
        adjust = weight[idx] / diag[idx]
        drop_preds = full_pred[:, None] - h * adjust[None, :]
        drop_ics = matrix_pred_xsz_ic(drop_preds, label, bounds)
        for local, factor_idx in enumerate(idx):
            drop_ic = float(drop_ics[local])
            rows.append(
                {
                    "factor": features[factor_idx],
                    "base_ic": base_ic,
                    "drop_ic": drop_ic,
                    "delta_ic": float(base_ic - drop_ic),
                    "retained": bool(drop_ic < base_ic),
                }
            )
        print(f"[select-ridge] evaluated {end}/{len(features)}", flush=True)
    result = pd.DataFrame(rows).sort_values("delta_ic", ascending=False)
    result.to_csv(path, index=False)
    retained = result[result["retained"]]["factor"].astype(str).tolist()
    write_list(out_dir / "ridge_selected_features.txt", retained)
    write_json(
        out_dir / "selection_ridge_summary.json",
        {"base_ic": base_ic, "features": len(features), "retained": len(retained), "alpha": alpha},
    )
    return retained


def fit_lgb(x: np.ndarray, y: np.ndarray, args: argparse.Namespace):
    import lightgbm as lgb

    params = dict(
        n_estimators=args.lgb_estimators,
        learning_rate=args.lgb_lr,
        num_leaves=args.lgb_leaves,
        subsample=0.82,
        colsample_bytree=args.lgb_colsample,
        min_child_samples=args.lgb_min_child,
        reg_lambda=args.lgb_lambda,
        random_state=args.seed,
        n_jobs=args.threads,
        verbose=-1,
        force_col_wise=True,
    )
    model = lgb.LGBMRegressor(**params)
    model.fit(x, y)
    return model


def lgb_shuffle_select(
    train: pd.DataFrame,
    val: pd.DataFrame,
    features: list[str],
    out_dir: Path,
    args: argparse.Namespace,
) -> list[str]:
    path = out_dir / "selection_lgb_shuffle_2019q4.csv"
    if path.exists():
        result = pd.read_csv(path)
        return result[result["retained"]]["factor"].astype(str).tolist()
    x, y = x_y(train, features)
    model = fit_lgb(x, y, args)
    split = model.booster_.feature_importance("split")
    gain = model.booster_.feature_importance("gain")
    xv, label, bounds, _meta = val_arrays(val, features)
    base_pred = model.booster_.predict(xv, num_threads=args.threads)
    base_ic = pred_xsz_ic(base_pred, label, bounds)
    rng = np.random.default_rng(args.seed + 17)
    partial_path = path.with_suffix(".partial.csv")
    rows = []
    done: set[str] = set()
    if partial_path.exists():
        partial = pd.read_csv(partial_path)
        rows = partial.to_dict("records")
        done = {str(row["factor"]) for row in rows}
        print(f"[select-lgb] resuming partial rows={len(rows)}", flush=True)
    target_indices = list(range(len(features)))
    if args.max_shuffle_features > 0:
        ranked = np.argsort(-gain)
        target_indices = [int(i) for i in ranked[: args.max_shuffle_features]]
    for n, idx in enumerate(target_indices, 1):
        if features[idx] in done:
            continue
        if split[idx] <= 0:
            shuffled_ic = base_ic
            delta = 0.0
            retained = False
        else:
            original = xv[:, idx].copy()
            xv[:, idx] = rng.standard_normal(len(xv)).astype(np.float32)
            pred = model.booster_.predict(xv, num_threads=args.threads)
            xv[:, idx] = original
            shuffled_ic = pred_xsz_ic(pred, label, bounds)
            delta = float(base_ic - shuffled_ic)
            retained = bool(shuffled_ic < base_ic)
        rows.append(
            {
                "factor": features[idx],
                "base_ic": base_ic,
                "shuffled_ic": float(shuffled_ic),
                "delta_ic": float(delta),
                "retained": retained,
                "split_importance": int(split[idx]),
                "gain_importance": float(gain[idx]),
            }
        )
        if n % 50 == 0 or n == len(target_indices):
            pd.DataFrame(rows).to_csv(partial_path, index=False)
            print(f"[select-lgb] evaluated {len(rows)}/{len(target_indices)}", flush=True)
    result = pd.DataFrame(rows).sort_values("delta_ic", ascending=False)
    result.to_csv(path, index=False)
    retained = result[result["retained"]]["factor"].astype(str).tolist()
    if args.max_shuffle_features > 0:
        retained = list(dict.fromkeys(retained + [features[int(i)] for i in np.argsort(-gain)[: args.lgb_keep_top_gain]]))
    write_list(out_dir / "lgb_selected_features.txt", retained)
    write_json(
        out_dir / "selection_lgb_summary.json",
        {
            "base_ic": base_ic,
            "features": len(features),
            "evaluated": len(target_indices),
            "retained": len(retained),
            "zero_split_features": int((split == 0).sum()),
            "max_shuffle_features": args.max_shuffle_features,
        },
    )
    return retained


def fit_mlp_model(
    train: pd.DataFrame,
    features: list[str],
    fit_month: str,
    cfg: MLPTrainConfig,
):
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    x, y = x_y(train, features)
    mean, scale = standardize_fit(x)
    x = ((x - mean) / scale).astype(np.float32)
    work = train.loc[np.isfinite(train["label_xsz"].to_numpy(np.float32, copy=False))].copy()
    w = recency_weights(work, fit_month, cfg.half_life_months)
    if w is None:
        w = np.ones(len(y), dtype=np.float32)
    w = np.nan_to_num(w.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    w = np.maximum(w, 0.0)
    w = w / max(float(w.mean()), 1e-8)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(cfg.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(cfg.seed)
    model = make_mlp(x.shape[1], cfg.hidden, cfg.dropout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    ds = TensorDataset(torch.from_numpy(x), torch.from_numpy(y), torch.from_numpy(w.astype(np.float32)))
    loader = DataLoader(ds, batch_size=cfg.batch_size, shuffle=True, drop_last=False, num_workers=0, pin_memory=device.type == "cuda")
    model.train()
    for epoch in range(cfg.epochs):
        total = 0.0
        batches = 0
        for xb, yb, wb in loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            wb = wb.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            pred = model(xb).squeeze(-1)
            loss = weighted_loss(pred, yb, wb, "mse")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            total += float(loss.detach().cpu())
            batches += 1
        print(f"[fit-mlp] epoch={epoch + 1}/{cfg.epochs} loss={total / max(batches, 1):.6f}", flush=True)
    return model, mean, scale, device


def predict_mlp(model, x: np.ndarray, mean: np.ndarray, scale: np.ndarray, device, chunk_rows: int) -> np.ndarray:
    import torch

    x = ((x - mean) / scale).astype(np.float32)
    out = np.empty(len(x), dtype=np.float32)
    model.eval()
    with torch.no_grad():
        for start in range(0, len(x), chunk_rows):
            end = min(start + chunk_rows, len(x))
            xb = torch.from_numpy(x[start:end]).to(device, non_blocking=True)
            out[start:end] = model(xb).squeeze(-1).detach().cpu().numpy().astype(np.float32)
    return out


def mlp_shuffle_select(
    train: pd.DataFrame,
    val: pd.DataFrame,
    features: list[str],
    out_dir: Path,
    args: argparse.Namespace,
) -> list[str]:
    path = out_dir / "selection_mlp_shuffle_2019q4.csv"
    if path.exists():
        result = pd.read_csv(path)
        return result[result["retained"]]["factor"].astype(str).tolist()
    cfg = MLPTrainConfig(epochs=args.mlp_select_epochs, batch_size=args.mlp_batch_size, seed=args.seed)
    model, mean, scale, device = fit_mlp_model(train, features, "2019-10", cfg)
    xv, label, bounds, _meta = val_arrays(val, features)
    base_pred = predict_mlp(model, xv, mean, scale, device, args.mlp_predict_chunk)
    base_ic = pred_xsz_ic(base_pred, label, bounds)
    xz = ((xv - mean) / scale).astype(np.float32)
    del xv
    import torch

    x_tensor = torch.from_numpy(xz).to(device)
    rng = np.random.default_rng(args.seed + 29)
    rows = []
    target_indices = list(range(len(features)))
    if args.max_shuffle_features > 0:
        # For fast dry runs, evaluate features with largest abs covariance on validation.
        cov = np.abs(np.nan_to_num(xz, nan=0.0).T @ np.nan_to_num(label, nan=0.0))
        target_indices = [int(i) for i in np.argsort(-cov)[: args.max_shuffle_features]]
    for n, idx in enumerate(target_indices, 1):
        original = x_tensor[:, idx].clone()
        replacement = torch.from_numpy(rng.standard_normal(x_tensor.shape[0]).astype(np.float32)).to(device)
        x_tensor[:, idx] = replacement
        pred = np.empty(x_tensor.shape[0], dtype=np.float32)
        model.eval()
        with torch.no_grad():
            for start in range(0, x_tensor.shape[0], args.mlp_predict_chunk):
                end = min(start + args.mlp_predict_chunk, x_tensor.shape[0])
                pred[start:end] = model(x_tensor[start:end]).squeeze(-1).detach().cpu().numpy().astype(np.float32)
        x_tensor[:, idx] = original
        shuffled_ic = pred_xsz_ic(pred, label, bounds)
        delta = float(base_ic - shuffled_ic)
        rows.append(
            {
                "factor": features[idx],
                "base_ic": base_ic,
                "shuffled_ic": float(shuffled_ic),
                "delta_ic": delta,
                "retained": bool(shuffled_ic < base_ic),
            }
        )
        if n % 20 == 0 or n == len(target_indices):
            pd.DataFrame(rows).to_csv(path.with_suffix(".partial.csv"), index=False)
            print(f"[select-mlp] evaluated {n}/{len(target_indices)}", flush=True)
    result = pd.DataFrame(rows).sort_values("delta_ic", ascending=False)
    result.to_csv(path, index=False)
    retained = result[result["retained"]]["factor"].astype(str).tolist()
    write_list(out_dir / "mlp_selected_features.txt", retained)
    write_json(
        out_dir / "selection_mlp_summary.json",
        {"base_ic": base_ic, "features": len(features), "evaluated": len(target_indices), "retained": len(retained)},
    )
    return retained


def run_select(args: argparse.Namespace) -> None:
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    features = read_feature_list(args.feature_file)
    pred_start = args.rolling_pred_start
    pred_end = args.rolling_pred_end
    train_months = month_range("2018-01", prev_month(pred_start))
    val_months = month_range("2019-10", "2019-12")
    train = load_samples(args.sample_dir, train_months, features)
    val = load_samples(args.sample_dir, val_months, features)
    print(f"[select] train={len(train)} val={len(val)} features={len(features)}", flush=True)

    ridge_features = ridge_leave_one_select(train, val, features, out_dir, args.ridge_alpha, args.ridge_block_size)
    lgb_features = lgb_shuffle_select(train, val, features, out_dir, args)
    mlp_features = mlp_shuffle_select(train, val, features, out_dir, args)
    write_validation_predictions(train, val, ridge_features, lgb_features, mlp_features, out_dir, args)
    write_json(
        out_dir / "selection_summary.json",
        {
            "features_total": len(features),
            "ridge_retained": len(ridge_features),
            "lgb_retained": len(lgb_features),
            "mlp_retained": len(mlp_features),
            "selection_window": "2019Q4 sample OOS after 2018-2019Q3 train",
            "uses_2020_for_selection": False,
        },
    )


def write_validation_predictions(
    train: pd.DataFrame,
    val: pd.DataFrame,
    ridge_features: list[str],
    lgb_features: list[str],
    mlp_features: list[str],
    out_dir: Path,
    args: argparse.Namespace,
) -> None:
    val_dir = out_dir / "validation_parts"
    if all((val_dir / f"{name}_2019q4.parquet").exists() for name in ["ridge", "lgb", "mlp"]):
        print("[validation] existing 2019Q4 predictions found", flush=True)
        return

    xr, yr = x_y(train, ridge_features)
    ridge = fit_ridge(xr, yr, args.ridge_alpha)
    xv, _label, _bounds, meta = val_arrays(val, ridge_features)
    save_pred(meta, predict_ridge(ridge, xv), val_dir / "ridge_2019q4.parquet")
    del xr, yr, xv
    gc.collect()

    xl, yl = x_y(train, lgb_features)
    lgb_model = fit_lgb(xl, yl, args)
    xv, _label, _bounds, meta = val_arrays(val, lgb_features)
    pred_lgb = lgb_model.booster_.predict(xv, num_threads=args.threads)
    save_pred(meta, pred_lgb, val_dir / "lgb_2019q4.parquet")
    del xl, yl, xv, pred_lgb
    gc.collect()

    mlp_cfg = MLPTrainConfig(epochs=args.mlp_select_epochs, batch_size=args.mlp_batch_size, seed=args.seed)
    mlp_model, mean, scale, device = fit_mlp_model(train[["datetime", "label", "label_xsz"] + mlp_features], mlp_features, "2019-10", mlp_cfg)
    xv, _label, _bounds, meta = val_arrays(val, mlp_features)
    pred_mlp = predict_mlp(mlp_model, xv, mean, scale, device, args.mlp_predict_chunk)
    save_pred(meta, pred_mlp, val_dir / "mlp_2019q4.parquet")
    del xv, pred_mlp, mlp_model
    if str(device) == "cuda":
        import torch

        torch.cuda.empty_cache()
    gc.collect()


def selected_or_default(path: Path, fallback: list[str]) -> list[str]:
    if path.exists():
        values = [x.strip() for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]
        if values:
            return values
    return fallback


def read_full_month(cfg, expr_file: Path, month: str, features: list[str]) -> pd.DataFrame:
    matrix = FeatureMatrix(cfg, expr_file)
    df = matrix.read_month(month, features)
    return df.sort_values(["datetime", "symbol"]).reset_index(drop=True)


def save_pred(df: pd.DataFrame, pred: np.ndarray, path: Path) -> None:
    out = df[["symbol", "datetime", "label"]].copy()
    out["pred"] = pred.astype(np.float32)
    out = add_cross_sectional_norms(out, "pred")
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(path, index=False)


def run_train_predict(args: argparse.Namespace) -> None:
    cfg = load_config(str(ROOT / "configs/futures.yaml"))
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    all_features = read_feature_list(args.feature_file)
    ridge_features = selected_or_default(out_dir / "ridge_selected_features.txt", all_features)
    lgb_features = selected_or_default(out_dir / "lgb_selected_features.txt", all_features)
    mlp_features = selected_or_default(out_dir / "mlp_selected_features.txt", all_features)
    train = load_samples(args.sample_dir, month_range("2018-01", "2019-12"), list(dict.fromkeys(ridge_features + lgb_features + mlp_features)))
    print(
        f"[train-final] rows={len(train)} ridge={len(ridge_features)} lgb={len(lgb_features)} mlp={len(mlp_features)}",
        flush=True,
    )

    xr, yr = x_y(train, ridge_features)
    ridge = fit_ridge(xr, yr, args.ridge_alpha)
    np.savez(out_dir / "ridge_final_model.npz", **ridge)
    del xr, yr
    gc.collect()

    xl, yl = x_y(train, lgb_features)
    lgb_model = fit_lgb(xl, yl, args)
    lgb_model.booster_.save_model(str(out_dir / "lgb_final_model.txt"))
    del xl, yl
    gc.collect()

    mlp_cfg = MLPTrainConfig(epochs=args.mlp_epochs, batch_size=args.mlp_batch_size, seed=args.seed)
    mlp_model, mlp_mean, mlp_scale, mlp_device = fit_mlp_model(train[["datetime", "label", "label_xsz"] + mlp_features], mlp_features, "2020-01", mlp_cfg)
    del train
    gc.collect()

    months = month_range("2020-01", "2020-12")
    for month in months:
        need = list(dict.fromkeys(ridge_features + lgb_features + mlp_features))
        df = read_full_month(cfg, args.expression_file, month, need)
        print(f"[predict-final] {month} rows={len(df)} features={len(need)}", flush=True)
        save_pred(df, predict_ridge(ridge, scrub_matrix(df[ridge_features].to_numpy(np.float32, copy=False))), out_dir / "prediction_parts" / "ridge" / f"{month}.parquet")
        pred_lgb = lgb_model.booster_.predict(
            scrub_matrix(df[lgb_features].to_numpy(np.float32, copy=False)),
            num_threads=args.threads,
        )
        save_pred(df, pred_lgb, out_dir / "prediction_parts" / "lgb" / f"{month}.parquet")
        pred_mlp = predict_mlp(mlp_model, scrub_matrix(df[mlp_features].to_numpy(np.float32, copy=False)), mlp_mean, mlp_scale, mlp_device, args.mlp_predict_chunk)
        save_pred(df, pred_mlp, out_dir / "prediction_parts" / "mlp" / f"{month}.parquet")
        del df, pred_lgb, pred_mlp
        gc.collect()
    write_json(
        out_dir / "final_model_metadata.json",
        {
            "ridge_features": len(ridge_features),
            "lgb_features": len(lgb_features),
            "mlp_features": len(mlp_features),
            "mlp_config": asdict(mlp_cfg),
            "uses_2020_for_feature_selection": False,
        },
    )


def sample_from_full_month(df: pd.DataFrame, features: list[str], month: str, rows: int, seed: int) -> pd.DataFrame:
    g = df.groupby("datetime", sort=False)["label"]
    work = df[["symbol", "datetime", "label"] + features].dropna(subset=["label"]).copy()
    work["label_xsz"] = ((work["label"] - g.transform("mean")) / (g.transform("std") + 1e-9)).clip(-8, 8).astype(np.float32)
    if len(work) > rows:
        rng = np.random.default_rng(seed + int(month.replace("-", "")))
        idx = np.sort(rng.choice(len(work), rows, replace=False))
        work = work.iloc[idx].copy()
    return work[["symbol", "datetime", "label", "label_xsz"] + features]


def run_rolling_lgb(args: argparse.Namespace) -> None:
    cfg = load_config(str(ROOT / "configs/futures.yaml"))
    out_dir = args.out_dir
    feature_path = out_dir / "lgb_selected_features.txt"
    features = read_feature_list(feature_path if feature_path.exists() else args.feature_file)
    parts_dir = out_dir / "prediction_parts" / "rolling_lgb"
    parts_dir.mkdir(parents=True, exist_ok=True)

    pred_start = args.rolling_pred_start
    pred_end = args.rolling_pred_end
    train_months = month_range("2018-01", prev_month(pred_start))
    train_samples = [
        load_samples(args.sample_dir, [month], features)
        for month in train_months
    ]
    pred_months = month_range(pred_start, pred_end)
    for month in pred_months:
        out_file = parts_dir / f"{month}.parquet"
        if out_file.exists():
            print(f"[rolling-lgb] exists {month}", flush=True)
            # If we are resuming, still add this month to the rolling train set.
            if month < pred_end:
                df_existing = read_full_month(cfg, args.expression_file, month, features)
                train_samples.append(sample_from_full_month(df_existing, features, month, args.rolling_sample_rows, args.seed))
            continue
        train = pd.concat(train_samples, ignore_index=True)
        x, y = x_y(train, features, args.lgb_target_col)
        print(
            f"[rolling-lgb] fit {month} train_rows={len(train)} "
            f"features={len(features)} target={args.lgb_target_col}",
            flush=True,
        )
        model = fit_lgb(x, y, args)
        df = read_full_month(cfg, args.expression_file, month, features)
        pred = model.booster_.predict(scrub_matrix(df[features].to_numpy(np.float32, copy=False)), num_threads=args.threads)
        save_pred(df, pred, out_file)
        if month >= "2020-01":
            mic = compute_ic(pd.read_parquet(out_file, columns=["pred_xsz"])["pred_xsz"].to_numpy(), df["label"].to_numpy())
            print(f"[rolling-lgb] wrote {month} rows={len(df)} pred_xsz_ic={mic:.6f}", flush=True)
        train_samples.append(sample_from_full_month(df, features, month, args.rolling_sample_rows, args.seed))
        # Keep memory bounded to a rolling expanding sample list of compact frames.
        del train, x, y, model, df, pred
        gc.collect()

    rows = []
    for year, months in [("2019q4", month_range("2019-10", "2019-12")), ("2020", month_range("2020-01", "2020-12"))]:
        frames = [pd.read_parquet(parts_dir / f"{m}.parquet") for m in months]
        data = pd.concat(frames, ignore_index=True)
        monthly = period_ic(data, "pred_xsz", "M")
        rows.append(
            {
                "model": "rolling_lgb",
                "window": year,
                "pred_ic": compute_ic(data["pred"].to_numpy(), data["label"].to_numpy()),
                "pred_xsz_ic": compute_ic(data["pred_xsz"].to_numpy(), data["label"].to_numpy()),
                "pred_xrank_ic": compute_ic(data["pred_xrank"].to_numpy(), data["label"].to_numpy()),
                "monthly_mean_xsz": float(monthly.mean()),
                "monthly_ir_xsz": float(monthly.mean() / monthly.std()) if monthly.std() > 0 else float("nan"),
                "features": len(features),
                "target_col": args.lgb_target_col,
            }
        )
    pd.DataFrame(rows).to_csv(out_dir / "rolling_lgb_summary.csv", index=False)
    print(pd.DataFrame(rows).to_string(index=False), flush=True)


def run_rolling_ridge(args: argparse.Namespace) -> None:
    cfg = load_config(str(ROOT / "configs/futures.yaml"))
    out_dir = args.out_dir
    feature_path = out_dir / "ridge_selected_features.txt"
    features = read_feature_list(feature_path if feature_path.exists() else args.feature_file)
    parts_dir = out_dir / "prediction_parts" / "rolling_ridge"
    parts_dir.mkdir(parents=True, exist_ok=True)

    pred_start = args.rolling_pred_start
    pred_end = args.rolling_pred_end
    train_samples = [load_samples(args.sample_dir, [month], features) for month in month_range("2018-01", prev_month(pred_start))]
    pred_months = month_range(pred_start, pred_end)
    for month in pred_months:
        out_file = parts_dir / f"{month}.parquet"
        if out_file.exists():
            print(f"[rolling-ridge] exists {month}", flush=True)
            if month < pred_end:
                df_existing = read_full_month(cfg, args.expression_file, month, features)
                train_samples.append(sample_from_full_month(df_existing, features, month, args.rolling_sample_rows, args.seed))
            continue
        train = pd.concat(train_samples, ignore_index=True)
        print(f"[rolling-ridge] fit {month} train_rows={len(train)} features={len(features)}", flush=True)
        x, y = x_y(train, features, args.ridge_target_col)
        model = fit_ridge(x, y, args.ridge_alpha)
        df = read_full_month(cfg, args.expression_file, month, features)
        pred = predict_ridge(model, scrub_matrix(df[features].to_numpy(np.float32, copy=False)))
        save_pred(df, pred, out_file)
        if month >= "2020-01":
            written = pd.read_parquet(out_file, columns=["pred_xsz", "label"])
            mic = compute_ic(written["pred_xsz"].to_numpy(), written["label"].to_numpy())
            print(f"[rolling-ridge] wrote {month} rows={len(df)} pred_xsz_ic={mic:.6f}", flush=True)
        train_samples.append(sample_from_full_month(df, features, month, args.rolling_sample_rows, args.seed))
        del train, x, y, model, df, pred
        gc.collect()

    rows = []
    for year, months in [("2019q4", month_range("2019-10", "2019-12")), ("2020", month_range("2020-01", "2020-12"))]:
        data = pd.concat([pd.read_parquet(parts_dir / f"{m}.parquet") for m in months], ignore_index=True)
        monthly = period_ic(data, "pred_xsz", "M")
        rows.append(
            {
                "model": "rolling_ridge",
                "window": year,
                "pred_ic": compute_ic(data["pred"].to_numpy(), data["label"].to_numpy()),
                "pred_xsz_ic": compute_ic(data["pred_xsz"].to_numpy(), data["label"].to_numpy()),
                "pred_xrank_ic": compute_ic(data["pred_xrank"].to_numpy(), data["label"].to_numpy()),
                "monthly_mean_xsz": float(monthly.mean()),
                "monthly_ir_xsz": float(monthly.mean() / monthly.std()) if monthly.std() > 0 else float("nan"),
                "features": len(features),
                "target_col": args.ridge_target_col,
            }
        )
    pd.DataFrame(rows).to_csv(out_dir / "rolling_ridge_summary.csv", index=False)
    print(pd.DataFrame(rows).to_string(index=False), flush=True)


def run_rolling_mlp(args: argparse.Namespace) -> None:
    cfg = load_config(str(ROOT / "configs/futures.yaml"))
    out_dir = args.out_dir
    feature_path = out_dir / "mlp_selected_features.txt"
    features = read_feature_list(feature_path if feature_path.exists() else args.feature_file)
    parts_dir = out_dir / "prediction_parts" / "rolling_mlp"
    parts_dir.mkdir(parents=True, exist_ok=True)

    pred_start = args.rolling_pred_start
    pred_end = args.rolling_pred_end
    train_samples = [load_samples(args.sample_dir, [month], features) for month in month_range("2018-01", prev_month(pred_start))]
    pred_months = month_range(pred_start, pred_end)
    mlp_cfg = MLPTrainConfig(epochs=args.mlp_epochs, batch_size=args.mlp_batch_size, seed=args.seed)
    for i, month in enumerate(pred_months):
        out_file = parts_dir / f"{month}.parquet"
        if out_file.exists():
            print(f"[rolling-mlp] exists {month}", flush=True)
            if month < pred_end:
                df_existing = read_full_month(cfg, args.expression_file, month, features)
                train_samples.append(sample_from_full_month(df_existing, features, month, args.rolling_sample_rows, args.seed))
            continue
        train = pd.concat(train_samples, ignore_index=True)
        print(f"[rolling-mlp] fit {month} train_rows={len(train)} features={len(features)}", flush=True)
        local_cfg = MLPTrainConfig(
            epochs=args.mlp_epochs,
            batch_size=args.mlp_batch_size,
            seed=args.seed + i,
            hidden=192,
            dropout=0.12,
        )
        model, mean, scale, device = fit_mlp_model(train[["datetime", "label", "label_xsz"] + features], features, month, local_cfg)
        df = read_full_month(cfg, args.expression_file, month, features)
        pred = predict_mlp(model, scrub_matrix(df[features].to_numpy(np.float32, copy=False)), mean, scale, device, args.mlp_predict_chunk)
        save_pred(df, pred, out_file)
        if month >= "2020-01":
            written = pd.read_parquet(out_file, columns=["pred_xsz", "label"])
            mic = compute_ic(written["pred_xsz"].to_numpy(), written["label"].to_numpy())
            print(f"[rolling-mlp] wrote {month} rows={len(df)} pred_xsz_ic={mic:.6f}", flush=True)
        train_samples.append(sample_from_full_month(df, features, month, args.rolling_sample_rows, args.seed))
        del train, model, mean, scale, df, pred
        if str(device) == "cuda":
            import torch

            torch.cuda.empty_cache()
        gc.collect()

    rows = []
    for year, months in [("2019q4", month_range("2019-10", "2019-12")), ("2020", month_range("2020-01", "2020-12"))]:
        data = pd.concat([pd.read_parquet(parts_dir / f"{m}.parquet") for m in months], ignore_index=True)
        monthly = period_ic(data, "pred_xsz", "M")
        rows.append(
            {
                "model": "rolling_mlp",
                "window": year,
                "pred_ic": compute_ic(data["pred"].to_numpy(), data["label"].to_numpy()),
                "pred_xsz_ic": compute_ic(data["pred_xsz"].to_numpy(), data["label"].to_numpy()),
                "pred_xrank_ic": compute_ic(data["pred_xrank"].to_numpy(), data["label"].to_numpy()),
                "monthly_mean_xsz": float(monthly.mean()),
                "monthly_ir_xsz": float(monthly.mean() / monthly.std()) if monthly.std() > 0 else float("nan"),
                "features": len(features),
            }
        )
    pd.DataFrame(rows).to_csv(out_dir / "rolling_mlp_summary.csv", index=False)
    print(pd.DataFrame(rows).to_string(index=False), flush=True)


def run_validate_selected(args: argparse.Namespace) -> None:
    out_dir = args.out_dir
    all_features = read_feature_list(args.feature_file)
    ridge_features = selected_or_default(out_dir / "ridge_selected_features.txt", all_features)
    lgb_features = selected_or_default(out_dir / "lgb_selected_features.txt", all_features)
    mlp_features = selected_or_default(out_dir / "mlp_selected_features.txt", all_features)
    need = list(dict.fromkeys(ridge_features + lgb_features + mlp_features))
    train = load_samples(args.sample_dir, month_range("2018-01", "2019-09"), need)
    val = load_samples(args.sample_dir, month_range("2019-10", "2019-12"), need)
    print(
        f"[validate-selected] train={len(train)} val={len(val)} "
        f"ridge={len(ridge_features)} lgb={len(lgb_features)} mlp={len(mlp_features)}",
        flush=True,
    )
    write_validation_predictions(train, val, ridge_features, lgb_features, mlp_features, out_dir, args)


def load_pred_parts(out_dir: Path, name: str, months: list[str], col: str = "pred_xsz") -> pd.DataFrame:
    parts = []
    for month in months:
        path = out_dir / "prediction_parts" / name / f"{month}.parquet"
        df = pd.read_parquet(path, columns=["symbol", "datetime", "label", col])
        df = df.rename(columns={col: name})
        df["datetime"] = pd.to_datetime(df["datetime"])
        parts.append(df)
    base = pd.concat(parts, ignore_index=True)
    return base


def run_ensemble(args: argparse.Namespace) -> None:
    out_dir = args.out_dir
    months = month_range("2020-01", "2020-12")
    names = ["ridge", "lgb", "mlp"]
    val_base = None
    for name in names:
        path = out_dir / "validation_parts" / f"{name}_2019q4.parquet"
        if not path.exists():
            raise FileNotFoundError(f"missing clean validation prediction: {path}")
        cur = pd.read_parquet(path, columns=["symbol", "datetime", "label", "pred_xsz"])
        cur = cur.rename(columns={"pred_xsz": name})
        cur["datetime"] = pd.to_datetime(cur["datetime"])
        if val_base is None:
            val_base = cur
        else:
            val_base = val_base.merge(cur[["symbol", "datetime", name]], on=["symbol", "datetime"], how="inner")
    assert val_base is not None
    xv = val_base[names].to_numpy(np.float64)
    yv = val_base["label"].to_numpy(np.float64)
    val_mask = np.isfinite(yv) & np.all(np.isfinite(xv), axis=1)
    xv = xv[val_mask]
    yv = yv[val_mask]
    c = xv.T @ yv
    g = xv.T @ xv
    yy = float(yv @ yv)
    lower = np.zeros(len(names), dtype=np.float64)
    upper = np.ones(len(names), dtype=np.float64)
    w, val_fit_ic = fit_ic_weights_from_stats(c, g, yy, lower, upper)

    base = None
    for name in names:
        cur = load_pred_parts(out_dir, name, months)
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
    base["pred"] = base[names].to_numpy(np.float32) @ w.astype(np.float32)
    base = add_cross_sectional_norms(base, "pred")
    pred_ic = compute_ic(base["pred"].to_numpy(), base["label"].to_numpy())
    pred_xsz_ic = compute_ic(base["pred_xsz"].to_numpy(), base["label"].to_numpy())
    monthly = period_ic(base, "pred_xsz", "M")
    corr = base[names].corr()
    base[["symbol", "datetime", "label", "pred", "pred_xsz", "pred_xrank"]].to_parquet(out_dir / "ensemble_2020_clean.parquet", index=False)
    monthly.to_csv(out_dir / "ensemble_monthly_ic.csv")
    corr.to_csv(out_dir / "component_corr_2020.csv")
    pd.DataFrame(
        [
            {
                "model": "fu_newfactor_three_model_ensemble_2020_clean_2019q4_gate",
                "pred_ic_2020": pred_ic,
                "pred_xsz_ic_2020": pred_xsz_ic,
                "monthly_mean": float(monthly.mean()),
                "monthly_ir": float(monthly.mean() / monthly.std()) if monthly.std() > 0 else float("nan"),
                "fit_ic_2019q4_validation": val_fit_ic,
                "uses_2020_for_weights": False,
                **{f"w_{n}": float(v) for n, v in zip(names, w)},
            }
        ]
    ).to_csv(out_dir / "ensemble_summary.csv", index=False)
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(10, 4))
        monthly.plot(kind="bar", ax=ax)
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_title("FU new-factor three-model ensemble monthly IC")
        ax.set_ylabel("IC")
        fig.tight_layout()
        fig.savefig(out_dir / "ensemble_monthly_ic.png", dpi=160)
        plt.close(fig)
    except Exception as exc:
        print(f"[plot] skipped: {exc}", flush=True)
    print(pd.read_csv(out_dir / "ensemble_summary.csv").to_string(index=False), flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage",
        choices=["select", "validate", "train_predict", "rolling_ridge", "rolling_lgb", "rolling_mlp", "ensemble", "all"],
        default="all",
    )
    parser.add_argument("--feature-file", type=Path, default=DEFAULT_FEATURES)
    parser.add_argument("--expression-file", type=Path, default=DEFAULT_EXPR)
    parser.add_argument("--sample-dir", type=Path, default=DEFAULT_SAMPLE_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--seed", type=int, default=20260625)
    parser.add_argument("--threads", type=int, default=int(os.environ.get("LIGHTGBM_NUM_THREADS", "8")))
    parser.add_argument("--ridge-alpha", type=float, default=1.0)
    parser.add_argument("--ridge-block-size", type=int, default=32)
    parser.add_argument("--ridge-target-col", choices=["label", "label_xsz"], default="label_xsz")
    parser.add_argument("--lgb-estimators", type=int, default=260)
    parser.add_argument("--lgb-lr", type=float, default=0.04)
    parser.add_argument("--lgb-leaves", type=int, default=63)
    parser.add_argument("--lgb-min-child", type=int, default=120)
    parser.add_argument("--lgb-lambda", type=float, default=4.0)
    parser.add_argument("--lgb-colsample", type=float, default=0.65)
    parser.add_argument("--lgb-target-col", choices=["label", "label_xsz"], default="label_xsz")
    parser.add_argument("--lgb-keep-top-gain", type=int, default=500)
    parser.add_argument("--mlp-select-epochs", type=int, default=3)
    parser.add_argument("--mlp-epochs", type=int, default=5)
    parser.add_argument("--mlp-batch-size", type=int, default=8192)
    parser.add_argument("--mlp-predict-chunk", type=int, default=131072)
    parser.add_argument("--max-shuffle-features", type=int, default=0)
    parser.add_argument("--rolling-sample-rows", type=int, default=30000)
    parser.add_argument("--rolling-pred-start", type=str, default="2019-10")
    parser.add_argument("--rolling-pred-end", type=str, default="2020-12")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.stage in {"select", "all"}:
        run_select(args)
    if args.stage == "validate":
        run_validate_selected(args)
    if args.stage in {"train_predict", "all"}:
        run_train_predict(args)
    if args.stage == "rolling_ridge":
        run_rolling_ridge(args)
    if args.stage == "rolling_lgb":
        run_rolling_lgb(args)
    if args.stage == "rolling_mlp":
        run_rolling_mlp(args)
    if args.stage in {"ensemble", "all"}:
        run_ensemble(args)


if __name__ == "__main__":
    main()
