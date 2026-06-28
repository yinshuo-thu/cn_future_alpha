#!/usr/bin/env python3
"""Screen clean ridge/LGB/MLP single models on the full 1,144-factor panel.

Protocol:
  - feature screening uses 2018-01..2019-09 samples for training and
    2019Q4 samples for validation;
  - final 2020 evaluation is monthly train-before-test;
  - no full-size prediction parquet is written.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch


FACTOR_PATH = Path("/root/shared-nvme/feature_model/data_factors_big.parquet")
CACHE_DIR = Path("/root/autodl-tmp/quant/ML/strict_opt_results/factor_count/month_sample_cache_all1144")
OUT_DIR = Path("/root/autodl-tmp/quant/ML/core_single_model_screen_results")

TRAIN_START = pd.Timestamp("2018-01-01")
SCREEN_VAL_START = pd.Timestamp("2019-10-01")
TEST_START = pd.Timestamp("2020-01-01")
TEST_END = pd.Timestamp("2021-01-01")

META_COLS = {
    "symbol",
    "datetime",
    "label",
    "label_xsz",
    "is_long_break_before",
    "session_id",
    "close",
    "open",
    "high",
    "low",
    "volume",
    "amount",
    "oi",
}


@dataclass(frozen=True)
class EvalResult:
    model: str
    feature_set: str
    n_features: int
    ic_2020: float
    monthly_mean_2020: float
    monthly_std_2020: float
    monthly_ir_2020: float
    rows_2020: int


def set_seed(seed: int = 20260628) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def month_starts(start: pd.Timestamp, end: pd.Timestamp) -> list[pd.Timestamp]:
    return list(pd.date_range(start, end - pd.offsets.MonthBegin(1), freq="MS"))


def feature_cols() -> list[str]:
    names = pq.ParquetFile(CACHE_DIR / "2019-01.parquet").schema_arrow.names
    return [c for c in names if c not in META_COLS]


def scrub(x: np.ndarray) -> np.ndarray:
    arr = np.asarray(x)
    if not arr.flags.writeable:
        arr = arr.copy()
    return np.nan_to_num(arr, copy=False, nan=0.0, posinf=0.0, neginf=0.0)


def compute_ic(pred: np.ndarray, label: np.ndarray) -> float:
    p = np.asarray(pred, dtype=np.float64)
    y = np.asarray(label, dtype=np.float64)
    mask = np.isfinite(p) & np.isfinite(y)
    if int(mask.sum()) < 2:
        return float("nan")
    p = p[mask]
    y = y[mask]
    den = math.sqrt(float(np.mean(p * p) * np.mean(y * y)))
    if den <= 1e-18:
        return float("nan")
    return float(np.mean(p * y) / den)


def load_sample_months(start: pd.Timestamp, end: pd.Timestamp, cols: list[str]) -> pd.DataFrame:
    pieces = []
    read_cols = ["symbol", "datetime", "label", "label_xsz", *cols]
    for ms in month_starts(start, end):
        path = CACHE_DIR / f"{ms:%Y-%m}.parquet"
        pieces.append(pd.read_parquet(path, columns=read_cols))
    out = pd.concat(pieces, ignore_index=True)
    out["datetime"] = pd.to_datetime(out["datetime"])
    return out[out["label"].notna() & out["label_xsz"].notna()].reset_index(drop=True)


def cap_rows(df: pd.DataFrame, cap: int, seed: int) -> pd.DataFrame:
    if cap <= 0 or len(df) <= cap:
        return df.reset_index(drop=True)
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(df), cap, replace=False)
    return df.iloc[np.sort(idx)].reset_index(drop=True)


def x_y(df: pd.DataFrame, cols: list[str], target: str = "label_xsz") -> tuple[np.ndarray, np.ndarray]:
    x = scrub(df[cols].to_numpy(np.float32)).astype(np.float32, copy=False)
    y = scrub(df[target].to_numpy(np.float32)).astype(np.float32, copy=False)
    return x, y


def x_only(df: pd.DataFrame, cols: list[str]) -> np.ndarray:
    return scrub(df[cols].to_numpy(np.float32)).astype(np.float32, copy=False)


def standardize_fit(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = x.mean(axis=0, dtype=np.float64).astype(np.float32)
    std = (x.std(axis=0, dtype=np.float64) + 1e-6).astype(np.float32)
    return ((x - mean) / std).astype(np.float32, copy=False), mean, std


def standardize_apply(x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return ((x - mean) / std).astype(np.float32, copy=False)


def add_pred_views(pred: pd.DataFrame) -> pd.DataFrame:
    g = pred.groupby("datetime", sort=False)["pred"]
    pred["pred_xsz"] = ((pred["pred"] - g.transform("mean")) / (g.transform("std") + 1e-9)).astype(np.float32)
    pred["pred_xrank"] = (g.rank(pct=True) - 0.5).astype(np.float32)
    return pred


def fit_ridge_arrays(x: np.ndarray, y: np.ndarray, alpha: float = 80.0) -> dict[str, Any]:
    xs, mean, std = standardize_fit(x)
    y_mean = float(y.mean())
    yc = (y - y_mean).astype(np.float64)
    xd = xs.astype(np.float64, copy=False)
    xtx = xd.T @ xd
    xtx.flat[:: xtx.shape[0] + 1] += float(alpha)
    coef = np.linalg.solve(xtx, xd.T @ yc).astype(np.float32)
    return {"mean": mean, "std": std, "coef": coef, "intercept": y_mean}


def pred_ridge_arrays(model: dict[str, Any], x: np.ndarray) -> np.ndarray:
    xs = standardize_apply(x, model["mean"], model["std"])
    return (xs @ model["coef"] + float(model["intercept"])).astype(np.float32)


def fit_lgb_arrays(x: np.ndarray, y: np.ndarray, seed: int = 20260628) -> lgb.LGBMRegressor:
    model = lgb.LGBMRegressor(
        objective="regression",
        n_estimators=int(os.environ.get("CORE_LGB_ESTIMATORS", "220")),
        learning_rate=float(os.environ.get("CORE_LGB_LR", "0.035")),
        num_leaves=int(os.environ.get("CORE_LGB_LEAVES", "63")),
        min_child_samples=int(os.environ.get("CORE_LGB_MIN_CHILD", "120")),
        subsample=float(os.environ.get("CORE_LGB_SUBSAMPLE", "0.82")),
        colsample_bytree=float(os.environ.get("CORE_LGB_COLSAMPLE", "0.62")),
        reg_lambda=float(os.environ.get("CORE_LGB_L2", "6.0")),
        n_jobs=int(os.environ.get("CORE_LGB_JOBS", "8")),
        random_state=seed,
        verbose=-1,
        force_col_wise=True,
    )
    model.fit(x, y)
    return model


def pred_lgb_arrays(model: lgb.LGBMRegressor, x: np.ndarray) -> np.ndarray:
    return model.booster_.predict(x).astype(np.float32)


class TorchMLP(torch.nn.Module):
    def __init__(self, n_features: int, hidden: int) -> None:
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(n_features, hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def fit_mlp_arrays(x: np.ndarray, y: np.ndarray, seed: int = 20260628) -> dict[str, Any]:
    xs, mean, std = standardize_fit(x)
    hidden = int(os.environ.get("CORE_MLP_HIDDEN", "192"))
    epochs = int(os.environ.get("CORE_MLP_EPOCHS", "6"))
    batch = int(os.environ.get("CORE_MLP_BATCH", "8192"))
    lr = float(os.environ.get("CORE_MLP_LR", "0.001"))
    wd = float(os.environ.get("CORE_MLP_ALPHA", "0.0001"))
    use_cuda = torch.cuda.is_available() and os.environ.get("CORE_MLP_DEVICE", "cuda") != "cpu"
    device = torch.device("cuda" if use_cuda else "cpu")
    torch.manual_seed(seed)
    if use_cuda:
        torch.cuda.manual_seed_all(seed)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    model = TorchMLP(xs.shape[1], hidden).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    loss_fn = torch.nn.MSELoss()
    x_tensor = torch.from_numpy(xs).to(device)
    y_tensor = torch.from_numpy(y.astype(np.float32, copy=False)).to(device)
    rng = np.random.default_rng(seed)
    model.train()
    for epoch in range(epochs):
        perm = rng.permutation(len(xs))
        total_loss = 0.0
        total_n = 0
        for start in range(0, len(xs), batch):
            idx = torch.as_tensor(perm[start : start + batch], device=device, dtype=torch.long)
            xb = x_tensor.index_select(0, idx)
            yb = y_tensor.index_select(0, idx)
            opt.zero_grad(set_to_none=True)
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            opt.step()
            bs = int(len(idx))
            total_loss += float(loss.detach().cpu()) * bs
            total_n += bs
        if os.environ.get("CORE_MLP_VERBOSE", "0") == "1":
            print(
                f"[mlp-fit] features={xs.shape[1]} epoch={epoch + 1}/{epochs} loss={total_loss / max(1, total_n):.6f}",
                flush=True,
            )
    model.eval()
    return {"mean": mean, "std": std, "model": model, "device": str(device)}


def pred_mlp_standardized(model: dict[str, Any], xs: np.ndarray) -> np.ndarray:
    net = model["model"]
    device = torch.device(model["device"])
    batch = int(os.environ.get("CORE_MLP_PRED_BATCH", "65536"))
    preds = []
    with torch.inference_mode():
        for start in range(0, len(xs), batch):
            xb = torch.from_numpy(xs[start : start + batch]).to(device)
            preds.append(net(xb).detach().cpu().numpy().astype(np.float32, copy=False))
    return np.concatenate(preds) if preds else np.empty(0, dtype=np.float32)


def pred_mlp_arrays(model: dict[str, Any], x: np.ndarray) -> np.ndarray:
    xs = standardize_apply(x, model["mean"], model["std"])
    return pred_mlp_standardized(model, xs)


def model_fit_predict(kind: str, train: pd.DataFrame, val: pd.DataFrame, cols: list[str], seed: int) -> tuple[Any, np.ndarray, float]:
    xtr, ytr = x_y(train, cols)
    xv, _ = x_y(val, cols)
    if kind == "ridge":
        model = fit_ridge_arrays(xtr, ytr)
        pred = pred_ridge_arrays(model, xv)
    elif kind == "lgb":
        model = fit_lgb_arrays(xtr, ytr, seed=seed)
        pred = pred_lgb_arrays(model, xv)
    elif kind == "mlp":
        model = fit_mlp_arrays(xtr, ytr, seed=seed)
        pred = pred_mlp_arrays(model, xv)
    else:
        raise ValueError(kind)
    ic = compute_ic(pred, val["label"].to_numpy(np.float64))
    return model, pred, ic


def gaussian_importance(
    kind: str,
    model: Any,
    val: pd.DataFrame,
    cols: list[str],
    baseline_ic: float,
    seed: int,
    max_rows: int,
) -> pd.DataFrame:
    work = cap_rows(val, max_rows, seed + 17)
    x, _ = x_y(work, cols)
    y_raw = work["label"].to_numpy(np.float64)
    rng = np.random.default_rng(seed + 123)
    rows = []
    if kind == "mlp":
        mean = model["mean"]
        std = model["std"]
        xs = standardize_apply(x, mean, std)
        base_ic = compute_ic(pred_mlp_standardized(model, xs), y_raw)
        for j, col in enumerate(cols):
            old = xs[:, j].copy()
            raw = x[:, j]
            mu = float(raw.mean())
            sd = float(raw.std() + 1e-6)
            noise = rng.normal(mu, sd, size=len(x)).astype(np.float32)
            xs[:, j] = ((noise - mean[j]) / std[j]).astype(np.float32, copy=False)
            ic = compute_ic(pred_mlp_standardized(model, xs), y_raw)
            xs[:, j] = old
            rows.append({"feature": col, "ic_after_noise": ic, "delta_ic": base_ic - ic})
            if (j + 1) % 100 == 0:
                print(f"[importance][{kind}] {j + 1}/{len(cols)} base_ic={base_ic:.6f}", flush=True)
        out = pd.DataFrame(rows).sort_values("delta_ic", ascending=False)
        out.attrs["baseline_ic"] = float(baseline_ic)
        out.attrs["importance_base_ic"] = float(base_ic)
        return out
    elif kind == "ridge":
        pred_fn = lambda arr: pred_ridge_arrays(model, arr)
    else:
        pred_fn = lambda arr: pred_lgb_arrays(model, arr)
    base_ic = compute_ic(pred_fn(x), y_raw)
    for j, col in enumerate(cols):
        old = x[:, j].copy()
        mu = float(old.mean())
        sd = float(old.std() + 1e-6)
        x[:, j] = rng.normal(mu, sd, size=len(x)).astype(np.float32)
        ic = compute_ic(pred_fn(x), y_raw)
        x[:, j] = old
        rows.append({"feature": col, "ic_after_noise": ic, "delta_ic": base_ic - ic})
        if (j + 1) % 100 == 0:
            print(f"[importance][{kind}] {j + 1}/{len(cols)} base_ic={base_ic:.6f}", flush=True)
    out = pd.DataFrame(rows).sort_values("delta_ic", ascending=False)
    out.attrs["baseline_ic"] = float(baseline_ic)
    out.attrs["importance_base_ic"] = float(base_ic)
    return out


def keep_from_importance(imp: pd.DataFrame, min_keep: int, max_keep: int) -> list[str]:
    keep = imp[imp["delta_ic"] > 0.0]["feature"].astype(str).tolist()
    if len(keep) < min_keep:
        keep = imp.head(min(min_keep, len(imp)))["feature"].astype(str).tolist()
    if len(keep) > max_keep:
        keep = imp.head(max_keep)["feature"].astype(str).tolist()
    return keep


def read_feature_list(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def ridge_stepwise(train: pd.DataFrame, val: pd.DataFrame, all_cols: list[str]) -> tuple[list[str], pd.DataFrame]:
    xtr, ytr = x_y(train, all_cols)
    xv, _ = x_y(val, all_cols)
    yv_raw = val["label"].to_numpy(np.float64)
    xtr_s, mean, std = standardize_fit(xtr)
    xv_s = standardize_apply(xv, mean, std)
    corr = np.nan_to_num((xtr_s.astype(np.float64).T @ ytr.astype(np.float64)) / max(1, len(ytr)))
    order = np.argsort(-np.abs(corr))
    grid = [50, 100, 150, 200, 300, 400, 600, 800, 1000, len(all_cols)]
    rows = []
    best_ic = -np.inf
    best_k = grid[0]
    for k in grid:
        k = min(k, len(all_cols))
        idx = order[:k]
        model = fit_ridge_arrays(xtr[:, idx], ytr)
        pred = pred_ridge_arrays(model, xv[:, idx])
        ic = compute_ic(pred, yv_raw)
        rows.append({"k": k, "val_ic_2019q4": ic})
        print(f"[ridge-stepwise] k={k} val_ic={ic:.6f}", flush=True)
        if np.isfinite(ic) and ic > best_ic:
            best_ic = ic
            best_k = k
    selected = [all_cols[i] for i in order[:best_k]]
    return selected, pd.DataFrame(rows)


def screen_kind(kind: str, all_cols: list[str], train: pd.DataFrame, val: pd.DataFrame) -> dict[str, list[str]]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if os.environ.get("CORE_REUSE_SCREEN", "0") == "1":
        existing = {
            f"round{round_id}": OUT_DIR / f"{kind}_round{round_id}_features.txt"
            for round_id in [1, 2]
        }
        if all(path.exists() for path in existing.values()):
            print(f"[screen][{kind}] reusing existing feature lists", flush=True)
            return {tag: read_feature_list(path) for tag, path in existing.items()}
    result: dict[str, list[str]] = {}
    if kind == "ridge":
        current, grid = ridge_stepwise(train, val, all_cols)
        grid.to_csv(OUT_DIR / "ridge_stepwise_grid.csv", index=False)
    else:
        current = list(all_cols)
    for round_id in [1, 2]:
        print(f"[screen][{kind}] round={round_id} features={len(current)}", flush=True)
        model, _pred, val_ic = model_fit_predict(kind, train, val, current, seed=20260628 + round_id)
        imp = gaussian_importance(
            kind,
            model,
            val,
            current,
            baseline_ic=val_ic,
            seed=20260628 + round_id,
            max_rows=int(os.environ.get("CORE_IMPORTANCE_ROWS", "8192")),
        )
        imp.to_csv(OUT_DIR / f"{kind}_round{round_id}_importance.csv", index=False)
        min_keep = int(os.environ.get(f"CORE_{kind.upper()}_MIN_KEEP", "60" if kind != "ridge" else "80"))
        max_keep = int(os.environ.get(f"CORE_{kind.upper()}_MAX_KEEP", "360" if kind != "ridge" else "500"))
        kept = keep_from_importance(imp, min_keep=min_keep, max_keep=max_keep)
        Path(OUT_DIR / f"{kind}_round{round_id}_features.txt").write_text("\n".join(kept) + "\n", encoding="utf-8")
        meta = {
            "kind": kind,
            "round": round_id,
            "features_before": len(current),
            "features_after": len(kept),
            "val_ic_2019q4": val_ic,
            "importance_rows": int(os.environ.get("CORE_IMPORTANCE_ROWS", "8192")),
        }
        (OUT_DIR / f"{kind}_round{round_id}_metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        result[f"round{round_id}"] = kept
        current = kept
        del model
        if kind == "mlp" and torch.cuda.is_available():
            torch.cuda.empty_cache()
    return result


def read_month_full(ms: pd.Timestamp, cols: list[str]) -> pd.DataFrame:
    next_ms = ms + pd.DateOffset(months=1)
    df = pd.read_parquet(
        FACTOR_PATH,
        columns=["symbol", "datetime", "label", *cols],
        filters=[("datetime", ">=", ms), ("datetime", "<", next_ms)],
    )
    df["datetime"] = pd.to_datetime(df["datetime"])
    return df.sort_values(["datetime", "symbol"], kind="mergesort").reset_index(drop=True)


def final_eval_2020(kind: str, tag: str, cols: list[str]) -> EvalResult:
    rows = []
    sums = {"py": 0.0, "p2": 0.0, "y2": 0.0, "n": 0}
    for ms in month_starts(TEST_START, TEST_END):
        train = load_sample_months(TRAIN_START, ms, cols)
        train = cap_rows(train, int(os.environ.get("CORE_FINAL_TRAIN_ROWS", "300000")), seed=8023 + ms.month)
        test = read_month_full(ms, cols)
        xtr, ytr = x_y(train, cols)
        xt = x_only(test, cols)
        if kind == "ridge":
            model = fit_ridge_arrays(xtr, ytr)
            pred = pred_ridge_arrays(model, xt)
        elif kind == "lgb":
            model = fit_lgb_arrays(xtr, ytr, seed=7300 + ms.month)
            pred = pred_lgb_arrays(model, xt)
        elif kind == "mlp":
            model = fit_mlp_arrays(xtr, ytr, seed=9100 + ms.month)
            pred = pred_mlp_arrays(model, xt)
        else:
            raise ValueError(kind)
        label = test["label"].to_numpy(np.float64)
        mask = np.isfinite(pred) & np.isfinite(label)
        p = pred[mask].astype(np.float64)
        y = label[mask]
        sums["py"] += float(p @ y)
        sums["p2"] += float(p @ p)
        sums["y2"] += float(y @ y)
        sums["n"] += int(mask.sum())
        month_ic = compute_ic(pred, label)
        rows.append({"model": kind, "feature_set": tag, "month": f"{ms:%Y-%m}", "rows": int(mask.sum()), "ic": month_ic})
        print(f"[final][{kind}][{tag}][{ms:%Y-%m}] features={len(cols)} rows={int(mask.sum())} ic={month_ic:.6f}", flush=True)
        del train, test, xtr, xt, model
        if kind == "mlp" and torch.cuda.is_available():
            torch.cuda.empty_cache()
    monthly = pd.DataFrame(rows)
    monthly.to_csv(OUT_DIR / f"{kind}_{tag}_monthly_ic.csv", index=False)
    n = max(1, sums["n"])
    ic = (sums["py"] / n) / math.sqrt(max((sums["p2"] / n) * (sums["y2"] / n), 1e-30))
    mean = float(monthly["ic"].mean())
    std = float(monthly["ic"].std(ddof=1))
    ir = mean / std if std > 0 else float("nan")
    return EvalResult(kind, tag, len(cols), float(ic), mean, std, ir, int(sums["n"]))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", default="ridge,lgb,mlp")
    parser.add_argument("--skip-final", action="store_true")
    args = parser.parse_args()
    set_seed()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    all_cols = feature_cols()
    print(f"[setup] features={len(all_cols)} out={OUT_DIR}", flush=True)
    train = load_sample_months(TRAIN_START, SCREEN_VAL_START, all_cols)
    val = load_sample_months(SCREEN_VAL_START, TEST_START, all_cols)
    train = cap_rows(train, int(os.environ.get("CORE_SCREEN_TRAIN_ROWS", "240000")), seed=101)
    val = cap_rows(val, int(os.environ.get("CORE_SCREEN_VAL_ROWS", "60000")), seed=102)
    print(f"[setup] screen_train={len(train)} screen_val={len(val)}", flush=True)
    summary_path = OUT_DIR / "final_summary.csv"
    summary_rows = []
    if summary_path.exists():
        summary_rows = pd.read_csv(summary_path).to_dict("records")
    for kind in [x.strip() for x in args.models.split(",") if x.strip()]:
        selected = screen_kind(kind, all_cols, train, val)
        if args.skip_final:
            continue
        evals = []
        for tag, cols in selected.items():
            evals.append(final_eval_2020(kind, tag, cols))
        best = max(evals, key=lambda r: r.ic_2020)
        Path(OUT_DIR / f"{kind}_selected_features.txt").write_text(
            "\n".join(selected[best.feature_set]) + "\n", encoding="utf-8"
        )
        for ev in evals:
            row = ev.__dict__.copy()
            row["selected"] = ev.feature_set == best.feature_set
            summary_rows.append(row)
        pd.DataFrame(summary_rows).drop_duplicates(["model", "feature_set"], keep="last").to_csv(summary_path, index=False)
    if summary_rows:
        print(pd.DataFrame(summary_rows).drop_duplicates(["model", "feature_set"], keep="last").to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
