from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from effective_rolling_single_models import (
    OUT_DIR,
    add_prediction_views,
    build_feature_matrix,
    cache_dir_for,
    compute_ic,
    ensure_month_cache,
    load_feature_sets,
    load_train_samples,
    month_range,
    read_list,
    recency_weights,
    summarize_predictions,
    transform_features,
    write_json,
    write_parquet_atomic,
)


class MLP(nn.Module):
    def __init__(self, n_features: int, hidden: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.LayerNorm(hidden // 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def scrub(x: np.ndarray) -> np.ndarray:
    arr = np.array(x, dtype=np.float32, copy=True)
    return np.nan_to_num(arr, copy=False, nan=0.0, posinf=0.0, neginf=0.0)


def weighted_mean_scale(x: np.ndarray, w: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    w64 = w.astype(np.float64)
    w64 = w64 / max(float(w64.sum()), 1e-12)
    mean = (w64[:, None] * x.astype(np.float64)).sum(axis=0).astype(np.float32)
    var = (w64[:, None] * (x.astype(np.float64) - mean) ** 2).sum(axis=0)
    scale = np.maximum(np.sqrt(var), 1e-6).astype(np.float32)
    return mean, scale


def weighted_loss(pred: torch.Tensor, y: torch.Tensor, w: torch.Tensor, loss_name: str) -> torch.Tensor:
    if loss_name == "mse":
        return ((pred - y) ** 2 * w).mean()
    if loss_name == "huber":
        loss = nn.functional.smooth_l1_loss(pred, y, reduction="none", beta=1.0)
        return (loss * w).mean()
    if loss_name == "corr_mse":
        mse = ((pred - y) ** 2 * w).mean()
        ws = w / torch.clamp(w.sum(), min=1e-8)
        px = pred - (ws * pred).sum()
        yx = y - (ws * y).sum()
        cov = (ws * px * yx).sum()
        var_p = torch.clamp((ws * px * px).sum(), min=1e-8)
        var_y = torch.clamp((ws * yx * yx).sum(), min=1e-8)
        corr_loss = 1.0 - cov / torch.sqrt(var_p * var_y)
        return 0.65 * mse + 0.35 * corr_loss
    raise ValueError(f"bad loss: {loss_name}")


def prediction_xsz_ic(pred: np.ndarray, frame: pd.DataFrame) -> float:
    tmp = frame[["datetime", "label"]].copy()
    tmp["pred"] = pred.astype(np.float32)
    return compute_ic(add_prediction_views(tmp)["pred_xsz"], tmp["label"])


def fit_predict_month(
    train: pd.DataFrame,
    test: pd.DataFrame,
    features: list[str],
    target_col: str,
    test_month: str,
    hidden: int,
    dropout: float,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    half_life_months: float,
    loss_name: str,
    select_top_k: int,
    val_months: int,
    patience: int,
    min_epochs: int,
    standardize: str,
    seed: int,
    device: torch.device,
) -> np.ndarray:
    y_all = train[target_col].to_numpy(np.float32, copy=False)
    mask = np.isfinite(y_all)
    work = train.loc[mask].copy()
    y_all = y_all[mask]
    w = recency_weights(train, test_month, half_life_months)
    if w is not None:
        w = w[mask].astype(np.float32)
        w = np.nan_to_num(w, nan=0.0, posinf=0.0, neginf=0.0)
        w = np.maximum(w, 0.0)
        w = w / max(float(w.mean()), 1e-8)
    else:
        w = np.ones(len(y_all), dtype=np.float32)

    fit_mask = np.ones(len(work), dtype=bool)
    val_mask = np.zeros(len(work), dtype=bool)
    if val_months > 0:
        test_period = pd.Period(test_month, freq="M")
        periods = work["datetime"].dt.to_period("M")
        val_start = test_period - val_months
        val_mask = (periods >= val_start) & (periods < test_period)
        if int(val_mask.sum()) >= 10_000 and int((~val_mask).sum()) >= 50_000:
            fit_mask = ~np.asarray(val_mask)
            val_mask = np.asarray(val_mask)
        else:
            fit_mask = np.ones(len(work), dtype=bool)
            val_mask = np.zeros(len(work), dtype=bool)

    x = scrub(work.loc[fit_mask, features].to_numpy(np.float32, copy=False))
    y = y_all[fit_mask]
    fit_w = w[fit_mask]

    if standardize == "weighted":
        mean, scale = weighted_mean_scale(x, fit_w)
    elif standardize == "unweighted":
        mean = x.mean(axis=0, dtype=np.float64).astype(np.float32)
        scale = np.maximum(x.std(axis=0, dtype=np.float64), 1e-6).astype(np.float32)
    else:
        raise ValueError(f"bad standardize: {standardize}")
    x = ((x - mean) / scale).astype(np.float32)
    selected_idx: np.ndarray | None = None
    if select_top_k > 0 and select_top_k < len(features):
        ws = fit_w.astype(np.float64)
        ws = ws / max(float(ws.sum()), 1e-12)
        y0 = y.astype(np.float64) - float(np.sum(ws * y.astype(np.float64)))
        cov = np.abs((x.astype(np.float64) * ws[:, None]).T @ y0)
        selected_idx = np.sort(np.argpartition(cov, -select_top_k)[-select_top_k:])
        x = x[:, selected_idx]

    y = np.clip(y, -8, 8).astype(np.float32)
    fit_w = fit_w.astype(np.float32)

    x_val = y_val = w_val = val_frame = None
    if val_mask.any():
        x_val = scrub(work.loc[val_mask, features].to_numpy(np.float32, copy=False))
        x_val = ((x_val - mean) / scale).astype(np.float32)
        if selected_idx is not None:
            x_val = x_val[:, selected_idx]
        y_val = np.clip(y_all[val_mask], -8, 8).astype(np.float32)
        w_val = w[val_mask].astype(np.float32)
        val_frame = work.loc[val_mask, ["datetime", "label"]].copy()

    torch.manual_seed(seed)
    model = MLP(x.shape[1], hidden, dropout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    ds = TensorDataset(torch.from_numpy(x), torch.from_numpy(y), torch.from_numpy(fit_w))
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=False, num_workers=0, pin_memory=(device.type == "cuda"))
    best_state = None
    best_score = -np.inf
    bad_epochs = 0
    model.train()
    for epoch in range(epochs):
        for xb, yb, wb in loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            wb = wb.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            pred = model(xb)
            loss = weighted_loss(pred, yb, wb, loss_name)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
        if x_val is not None:
            model.eval()
            preds = []
            with torch.no_grad():
                for start in range(0, len(x_val), 200_000):
                    xb = torch.from_numpy(x_val[start : start + 200_000]).to(device)
                    preds.append(model(xb).detach().cpu().numpy().astype(np.float32))
            val_pred = np.concatenate(preds) if preds else np.empty(0, dtype=np.float32)
            score = prediction_xsz_ic(val_pred, val_frame)
            if score > best_score:
                best_score = score
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                bad_epochs = 0
            else:
                bad_epochs += 1
            model.train()
            if epoch + 1 >= min_epochs and bad_epochs >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)

    pred_out = np.empty(len(test), dtype=np.float32)
    model.eval()
    with torch.no_grad():
        for start in range(0, len(test), 200_000):
            end = min(start + 200_000, len(test))
            xt = scrub(test.iloc[start:end][features].to_numpy(np.float32, copy=False))
            xt = ((xt - mean) / scale).astype(np.float32)
            if selected_idx is not None:
                xt = xt[:, selected_idx]
            pt = model(torch.from_numpy(xt).to(device)).detach().cpu().numpy().astype(np.float32)
            pred_out[start:end] = pt
    del model, opt, loader, ds, x, y, w, work
    if device.type == "cuda":
        torch.cuda.empty_cache()
    gc.collect()
    return pred_out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature-set", choices=["ridge617", "lgbm643", "overlap333", "union927"], default="ridge617")
    parser.add_argument("--feature-file", type=Path, default=None)
    parser.add_argument("--name", default="mlp_ridge617_xsz_hl12_n400k")
    parser.add_argument("--train-start", default="2017-01")
    parser.add_argument("--test-start", default="2019-01")
    parser.add_argument("--test-end", default="2020-12")
    parser.add_argument("--cache-rows-per-month", type=int, default=30_000)
    parser.add_argument("--max-train-rows", type=int, default=400_000)
    parser.add_argument("--embargo-bars", type=int, default=30)
    parser.add_argument("--half-life-months", type=float, default=12.0)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.12)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--feature-transform", choices=["raw", "xsz"], default="raw")
    parser.add_argument("--target-col", choices=["label_xsz", "label_xrank", "label_ranknorm"], default="label_xsz")
    parser.add_argument("--loss", choices=["mse", "huber", "corr_mse"], default="mse")
    parser.add_argument("--select-top-k", type=int, default=0)
    parser.add_argument("--val-months", type=int, default=0)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--min-epochs", type=int, default=3)
    parser.add_argument("--standardize", choices=["unweighted", "weighted"], default="unweighted")
    parser.add_argument("--rebuild-cache", action="store_true")
    args = parser.parse_args()

    feature_sets = load_feature_sets()
    features = read_list(args.feature_file) if args.feature_file is not None else feature_sets[args.feature_set]
    fm = build_feature_matrix()
    train_months = month_range(args.train_start, str(pd.Period(args.test_end, freq="M") - 1))
    test_months = month_range(args.test_start, args.test_end)
    cache_dir = ensure_month_cache(
        fm,
        train_months,
        features,
        args.cache_rows_per_month,
        args.rebuild_cache,
        20260624,
        args.feature_transform,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    out_dir = OUT_DIR / args.name
    parts_dir = out_dir / "month_parts"
    parts_dir.mkdir(parents=True, exist_ok=True)
    for i, month in enumerate(test_months):
        part_path = parts_dir / f"{month}.parquet"
        if part_path.exists():
            continue
        train = load_train_samples(
            cache_dir,
            train_months,
            month,
            args.max_train_rows,
            "soft_event",
            20260624 + i,
            args.embargo_bars,
            0,
        )
        test = fm.read_month(month, features)
        test = transform_features(test, features, args.feature_transform)
        pred = fit_predict_month(
            train,
            test,
            features,
            args.target_col,
            month,
            args.hidden,
            args.dropout,
            args.epochs,
            args.batch_size,
            args.lr,
            args.weight_decay,
            args.half_life_months,
            args.loss,
            args.select_top_k,
            args.val_months,
            args.patience,
            args.min_epochs,
            args.standardize,
            20260624 + i,
            device,
        )
        out = test[["symbol", "datetime", "label"]].copy()
        out["pred"] = pred
        write_parquet_atomic(out, part_path)
        print(f"[mlp][{args.name}][{month}] train={len(train)} test={len(test)} ic={compute_ic(out['pred'], out['label']):.6f}", flush=True)
        del train, test, out, pred
        gc.collect()

    pred_df = pd.concat([pd.read_parquet(parts_dir / f"{month}.parquet") for month in test_months], ignore_index=True)
    pred_df = add_prediction_views(pred_df)
    write_parquet_atomic(pred_df, out_dir / f"{args.name}.parquet")
    row = summarize_predictions(pred_df, args.name)
    row.update({f"cfg_{k}": v for k, v in vars(args).items()})
    pd.DataFrame([row]).to_csv(out_dir / "summary.csv", index=False)
    monthly_rows = []
    for month, grp in pred_df.assign(month=pred_df["datetime"].dt.to_period("M").astype(str)).groupby("month", sort=True):
        monthly_rows.append(
            {
                "model": args.name,
                "month": month,
                "pred_ic": compute_ic(grp["pred"], grp["label"]),
                "pred_xsz_ic": compute_ic(grp["pred_xsz"], grp["label"]),
                "pred_xrank_ic": compute_ic(grp["pred_xrank"], grp["label"]),
            }
        )
    pd.DataFrame(monthly_rows).to_csv(out_dir / "monthly_ic.csv", index=False)
    metadata = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }
    write_json(out_dir / "metadata.json", metadata | {"device": str(device), "features": len(features)})
    print(f"[mlp][summary] {json.dumps(row, ensure_ascii=True)[:500]}", flush=True)


if __name__ == "__main__":
    main()
