#!/usr/bin/env python3
"""Rolling anchored deep MOE over strict OOS component predictions.

Selection is clean:
  - architecture/epoch/blend alpha are selected on 2019-Q4 only;
  - 2020 monthly models are refit using labels strictly before the test month;
  - 2020 labels are used only after predictions are written for reporting.
"""

from __future__ import annotations

import gc
import json
import os
import sys
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, "/root/autodl-tmp/quant/ML")

import numpy as np
import pandas as pd
import torch

from deep_moe_lowmem import (
    AnchoredDeepMOE,
    add_context,
    build_gate_features,
    config_from_env,
    ensure_component_memmap,
    fit_blend_alpha,
    fit_static_weights,
    make_label_xsz,
    make_loader,
    matvec_chunks,
    plot_monthly,
    predict_batches,
    sample_indices,
    set_seed,
    train_one_epoch,
)
from rolling_factor_model_eval import add_cross_sectional_norms, compute_ic, period_ic
from strict_optimization_ablation import OUT_DIR as STRICT_OUT_DIR
from strict_optimization_ablation import PRED_START, TEST_END, TEST_START, summarize


OUT_DIR = Path(os.environ.get("DEEP_ROLLING_OUT_DIR", str(STRICT_OUT_DIR / "deep_moe_rolling")))
VAL_START = pd.Timestamp("2019-10-01")


def fit_and_predict(
    *,
    x: np.ndarray,
    y: np.ndarray,
    target_all: np.ndarray,
    base: pd.DataFrame,
    names: list[str],
    symbol_code: np.ndarray,
    group_code: np.ndarray,
    cont: np.ndarray,
    train_mask: np.ndarray,
    test_idx: np.ndarray,
    cfg,
    device: torch.device,
    epochs: int,
    seed_offset: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, np.ndarray]:
    static_w, static_train_ic = fit_static_weights(x, y, train_mask, len(names))
    static_pred_all = matvec_chunks(x, static_w)
    train_idx = sample_indices(base, train_mask & np.isfinite(target_all), cfg.max_train_rows, cfg.seed + seed_offset)
    gate_x = build_gate_features(x, train_idx, cont, static_pred_all, cfg.gate_mode)
    model = AnchoredDeepMOE(len(names), gate_x.shape[1], int(symbol_code.max()) + 1, int(group_code.max()) + 1, static_w, cfg).to(device)
    loader = make_loader(x, gate_x, symbol_code, group_code, target_all, train_idx, cfg, shuffle=True)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    for epoch in range(1, epochs + 1):
        loss = train_one_epoch(model, loader, opt, cfg, device)
        print(f"[deep-rolling][fit] epoch={epoch}/{epochs} loss={loss:.6f} train_rows={len(train_idx)}", flush=True)
    dyn_pred, weight_mean, weight_std = predict_batches(model, x, gate_x, symbol_code, group_code, test_idx, device, cfg.batch_size * 2)
    static_test_pred = static_pred_all[test_idx]
    del gate_x, loader, opt, model, static_pred_all
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return dyn_pred, static_test_pred, weight_mean, static_train_ic, weight_std


def select_epoch_alpha(
    *,
    x: np.ndarray,
    y: np.ndarray,
    target_all: np.ndarray,
    base: pd.DataFrame,
    names: list[str],
    dt: pd.Series,
    symbol_code: np.ndarray,
    group_code: np.ndarray,
    cont: np.ndarray,
    cfg,
    device: torch.device,
) -> tuple[int, float, pd.DataFrame]:
    fit_mask = ((dt >= PRED_START) & (dt < VAL_START) & base["label"].notna()).to_numpy()
    val_mask = ((dt >= VAL_START) & (dt < TEST_START) & base["label"].notna()).to_numpy()
    val_idx = sample_indices(base, val_mask & np.isfinite(target_all), cfg.max_val_rows, cfg.seed + 1)

    static_w, static_train_ic = fit_static_weights(x, y, fit_mask, len(names))
    static_pred_all = matvec_chunks(x, static_w)
    train_idx = sample_indices(base, fit_mask & np.isfinite(target_all), cfg.max_train_rows, cfg.seed)
    gate_x = build_gate_features(x, train_idx, cont, static_pred_all, cfg.gate_mode)
    static_val_pred = static_pred_all[val_idx]
    val_label = y[val_idx]
    static_val_ic = compute_ic(static_val_pred, val_label)
    print(
        f"[deep-rolling][select] components={len(names)} train={len(train_idx)} val={len(val_idx)} "
        f"static_train_ic_2019q1q3={static_train_ic:.6f} static_val_ic_2019q4={static_val_ic:.6f}",
        flush=True,
    )

    model = AnchoredDeepMOE(len(names), gate_x.shape[1], int(symbol_code.max()) + 1, int(group_code.max()) + 1, static_w, cfg).to(device)
    loader = make_loader(x, gate_x, symbol_code, group_code, target_all, train_idx, cfg, shuffle=True)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    best_epoch = 0
    best_alpha = 0.0
    best_ic = -np.inf
    stale = 0
    rows = []
    for epoch in range(1, cfg.epochs + 1):
        loss = train_one_epoch(model, loader, opt, cfg, device)
        val_pred, _, _ = predict_batches(model, x, gate_x, symbol_code, group_code, val_idx, device, cfg.batch_size * 2)
        val_ic = compute_ic(val_pred, val_label)
        alpha, blend_ic = fit_blend_alpha(static_val_pred, val_pred, val_label, cfg)
        row = {
            "epoch": epoch,
            "loss": loss,
            "val_ic_2019q4": val_ic,
            "blend_ic_2019q4": blend_ic,
            "blend_alpha": alpha,
            "static_val_ic_2019q4": static_val_ic,
            "static_train_ic_2019q1q3": static_train_ic,
        }
        rows.append(row)
        print(
            f"[deep-rolling][select][epoch {epoch:02d}] loss={loss:.6f} val_ic={val_ic:.6f} "
            f"blend_ic={blend_ic:.6f} alpha={alpha:.3f}",
            flush=True,
        )
        if blend_ic > best_ic:
            best_ic = float(blend_ic)
            best_alpha = float(alpha)
            best_epoch = int(epoch)
            stale = 0
        else:
            stale += 1
            if stale >= cfg.early_stop_patience:
                break
    del gate_x, loader, opt, model, static_pred_all
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if best_epoch <= 0:
        raise RuntimeError("deep rolling selection failed")
    return best_epoch, best_alpha, pd.DataFrame(rows)


def main() -> None:
    cfg = config_from_env()
    set_seed(cfg.seed)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    parts_dir = OUT_DIR / "month_parts"
    parts_dir.mkdir(parents=True, exist_ok=True)

    base, names, x = ensure_component_memmap()
    y = base["label"].to_numpy(np.float64)
    dt = base["datetime"]
    label_xsz = make_label_xsz(base)
    if cfg.target_mode == "xsz":
        target_all = label_xsz
    elif cfg.target_mode == "raw":
        target_all = y.astype(np.float32)
    else:
        raise ValueError(f"unknown DEEP_LOWMEM_TARGET_MODE={cfg.target_mode!r}")
    symbol_code, group_code, cont, sym_map, grp_map = add_context(base)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    best_epoch, best_alpha, history = select_epoch_alpha(
        x=x,
        y=y,
        target_all=target_all,
        base=base,
        names=names,
        dt=dt,
        symbol_code=symbol_code,
        group_code=group_code,
        cont=cont,
        cfg=cfg,
        device=device,
    )
    history.to_csv(OUT_DIR / "selection_history.csv", index=False)
    print(f"[deep-rolling] selected_epoch={best_epoch} selected_alpha={best_alpha:.3f}", flush=True)

    weight_rows = []
    for ms in pd.date_range(TEST_START, TEST_END - pd.offsets.MonthBegin(1), freq="MS"):
        part_path = parts_dir / f"{ms:%Y-%m}.parquet"
        if part_path.exists():
            print(f"[deep-rolling][{ms:%Y-%m}] ckpt", flush=True)
            continue
        train_mask = ((dt >= PRED_START) & (dt < ms) & base["label"].notna()).to_numpy()
        test_idx = np.flatnonzero((dt >= ms) & (dt < ms + pd.DateOffset(months=1)))
        dyn_pred, static_pred, weight_mean, static_train_ic, weight_std = fit_and_predict(
            x=x,
            y=y,
            target_all=target_all,
            base=base,
            names=names,
            symbol_code=symbol_code,
            group_code=group_code,
            cont=cont,
            train_mask=train_mask,
            test_idx=test_idx,
            cfg=cfg,
            device=device,
            epochs=best_epoch,
            seed_offset=100 + int(ms.year * 12 + ms.month),
        )
        pred = static_pred + best_alpha * (dyn_pred - static_pred)
        out = base.iloc[test_idx][["symbol", "datetime", "label"]].copy()
        out["pred"] = pred.astype(np.float32)
        out.to_parquet(part_path, index=False)
        month_ic = compute_ic(out["pred"].to_numpy(), out["label"].to_numpy())
        row = {"month": f"{ms:%Y-%m}", "static_train_ic": static_train_ic, "test_ic": month_ic}
        row.update({f"wmean_{name}": float(v) for name, v in zip(names, weight_mean)})
        row.update({f"wstd_{name}": float(v) for name, v in zip(names, weight_std)})
        weight_rows.append(row)
        print(f"[deep-rolling][{ms:%Y-%m}] rows={len(out)} ic={month_ic:.6f}", flush=True)

    pieces = [pd.read_parquet(parts_dir / f"{ms:%Y-%m}.parquet") for ms in pd.date_range(TEST_START, TEST_END - pd.offsets.MonthBegin(1), freq="MS")]
    pred = pd.concat(pieces, ignore_index=True)
    pred = add_cross_sectional_norms(pred, "pred")
    pred.to_parquet(OUT_DIR / "deep_moe_rolling.parquet", index=False)
    summary = pd.DataFrame([summarize(pred, "deep_moe_rolling") | {"best_epoch_2019q4": best_epoch, "best_alpha_2019q4": best_alpha}])
    summary.to_csv(OUT_DIR / "summary.csv", index=False)
    monthly = period_ic(pred, "pred", "M")
    monthly.to_csv(OUT_DIR / "monthly_ic.csv")
    plot_monthly(monthly, OUT_DIR / "monthly_ic.png")
    if weight_rows:
        pd.DataFrame(weight_rows).to_csv(OUT_DIR / "monthly_weight_stats.csv", index=False)
    meta = {
        "config": asdict(cfg),
        "components": names,
        "selection_window": "2019-01..2019-09 train, 2019-10..2019-12 validate",
        "rolling_rule": "each 2020 month trains on rows before that month only",
        "device": str(device),
        "n_symbols": len(sym_map),
        "n_groups": len(grp_map),
    }
    (OUT_DIR / "metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(summary[["model", "pred_ic_2020", "pred_monthly_mean_2020", "pred_monthly_ir_2020", "best_alpha_2019q4"]].to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
