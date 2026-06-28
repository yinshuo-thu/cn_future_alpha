#!/usr/bin/env python3
"""Select factor columns using only 2018 labels.

This intentionally avoids any 2019/2020 information when ranking factors.  The
output is a clean feature list that can feed strict monthly base learners.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from strict_optimization_ablation import FACTOR_PATH, META_COLS, OUT_DIR


CATALOG_PATH = Path("/root/autodl-tmp/quant/artifacts/factor_catalog.csv")
OUT_CSV = OUT_DIR / "selected_factors_2018_ic.csv"
OUT_TXT = OUT_DIR / "selected_factors_2018_ic.txt"
START = pd.Timestamp("2018-01-01")
END = pd.Timestamp("2019-01-01")


def ic_vec(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    x = np.nan_to_num(x.astype(np.float64, copy=False), copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    c = x.T @ y
    g = np.einsum("ij,ij->j", x, x)
    yy = float(y @ y)
    den = np.sqrt(np.maximum(g * yy, 1e-24))
    return c / den


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pf = pq.ParquetFile(FACTOR_PATH)
    all_cols = pf.schema_arrow.names
    feature_cols = [c for c in all_cols if c not in set(META_COLS)]
    if CATALOG_PATH.exists():
        catalog = pd.read_csv(CATALOG_PATH)
        known = set(catalog["factor"].astype(str))
        feature_cols = [c for c in feature_cols if c in known]
    print(f"[select2018] features={len(feature_cols)} path={FACTOR_PATH}", flush=True)

    base = pd.read_parquet(
        FACTOR_PATH,
        columns=["datetime", "label"],
        filters=[("datetime", ">=", START), ("datetime", "<", END)],
    )
    base["datetime"] = pd.to_datetime(base["datetime"])
    good_label = base["label"].notna().to_numpy()
    g = base.groupby("datetime", sort=False)["label"]
    y = ((base["label"] - g.transform("mean")) / (g.transform("std") + 1e-9)).clip(-8, 8).to_numpy(np.float64)
    y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
    y[~good_label] = 0.0
    months = base["datetime"].dt.to_period("M").astype(str).to_numpy()
    month_values = sorted(pd.unique(months))
    month_idx = [np.flatnonzero(months == m) for m in month_values]
    print(f"[select2018] rows={len(base)} months={month_values[0]}..{month_values[-1]}", flush=True)

    chunk_size = int(os.environ.get("SELECT2018_CHUNK_SIZE", "48"))
    rows: list[dict[str, object]] = []
    for start in range(0, len(feature_cols), chunk_size):
        chunk = feature_cols[start : start + chunk_size]
        df = pd.read_parquet(
            FACTOR_PATH,
            columns=chunk,
            filters=[("datetime", ">=", START), ("datetime", "<", END)],
        )
        x = df.to_numpy(np.float32, copy=False)
        x = np.nan_to_num(x, copy=True, nan=0.0, posinf=0.0, neginf=0.0)
        x[~good_label, :] = 0.0
        overall = ic_vec(x, y)
        monthly = []
        for idx in month_idx:
            monthly.append(ic_vec(x[idx], y[idx]))
        monthly_arr = np.vstack(monthly)
        mean_ic = monthly_arr.mean(axis=0)
        abs_mean_ic = np.abs(monthly_arr).mean(axis=0)
        std_ic = monthly_arr.std(axis=0)
        sign_ref = np.sign(mean_ic)
        sign_ref[sign_ref == 0] = np.sign(overall[sign_ref == 0])
        sign_ref[sign_ref == 0] = 1.0
        hit = (np.sign(monthly_arr) == sign_ref[None, :]).mean(axis=0)
        score = np.abs(mean_ic) + 0.25 * np.abs(overall) + 0.10 * abs_mean_ic + 0.02 * hit - 0.05 * std_ic
        for j, name in enumerate(chunk):
            rows.append(
                {
                    "factor": name,
                    "score_2018": float(score[j]),
                    "overall_ic_2018": float(overall[j]),
                    "monthly_mean_ic_2018": float(mean_ic[j]),
                    "monthly_abs_mean_ic_2018": float(abs_mean_ic[j]),
                    "monthly_std_ic_2018": float(std_ic[j]),
                    "monthly_sign_hit_2018": float(hit[j]),
                }
            )
        print(f"[select2018] {min(start + len(chunk), len(feature_cols))}/{len(feature_cols)}", flush=True)
        del df, x

    out = pd.DataFrame(rows)
    if CATALOG_PATH.exists():
        out = out.merge(pd.read_csv(CATALOG_PATH), on="factor", how="left")
    out = out.sort_values("score_2018", ascending=False).reset_index(drop=True)
    out.to_csv(OUT_CSV, index=False)
    OUT_TXT.write_text("\n".join(out["factor"].astype(str).tolist()) + "\n", encoding="utf-8")
    print(out.head(40).to_string(index=False), flush=True)
    print(f"[select2018] wrote {OUT_CSV} and {OUT_TXT}", flush=True)


if __name__ == "__main__":
    main()
