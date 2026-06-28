"""
Build the large factor checkpoint — MEMORY-BOUNDED for a 100GB cgroup.

Key memory tactics:
  - float32 everywhere; downcast per-symbol frames before concat.
  - row-filter to FACTOR_START (Plan A predicts 2018+, 12m lookback -> keep 2017+).
  - derived views (tsz / csz / csr) computed ONE AT A TIME; only a row-sample is
    retained for correlation dedup, the full view is freed immediately.
  - materialize ONLY the selected factors (recompute selected derived columns).

Pipeline: raw(+amt/OI) -> sessions -> labels -> TS factors -> row filter ->
sample-based 0.9 correlation dedup over 4 views -> single-factor IC screen ->
materialize selected -> outputs/data_factors_big.parquet (+ selected_factors.txt)
"""
import os
import sys
sys.path.insert(0, "/root/feature_model")
import gc
import time
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from src.data.loader import load_all, load_config
from src.data.sessions import add_sessions_all
from src.data.labels import build_labels_all
from src.features.factor_lib import compute_symbol_factors
from src.evaluation.metrics import compute_ic
from src.journal import init_journal, log_attempt, log_result, log_shared_conclusion

CORR_THRESH = 0.9
TARGET_MIN = 1000
FACTOR_START = "2017-01-01"   # keep >= this (2018 predictions need 12m lookback)
SAMPLE_N = 200000
BIG_DIR = "/root/shared-nvme/feature_model"   # 50GB NVMe; /root overlay is only ~20GB free
META = ["symbol", "datetime", "label", "is_long_break_before", "session_id",
        "close", "open", "high", "low", "volume"]


def _csz(panel, cols):
    g = panel.groupby("datetime", sort=False)
    out = {}
    for c in cols:
        mu = g[c].transform("mean")
        sd = g[c].transform("std")
        out[c] = ((panel[c] - mu) / (sd + 1e-8)).astype(np.float32)
    return pd.DataFrame(out, index=panel.index)


def _csr(panel, cols):
    return (panel.groupby("datetime", sort=False)[cols].rank(pct=True) - 0.5).astype(np.float32)


def _tsz(panel, cols):
    window = 120
    minp = 30
    out = np.full((len(panel), len(cols)), np.nan, dtype=np.float32)
    for _, idx in panel.groupby("symbol", sort=False).indices.items():
        pos = np.asarray(idx, dtype=np.int64)
        vals = panel.iloc[pos][cols].to_numpy(dtype=np.float32, copy=True)
        finite = np.isfinite(vals)
        clean = np.where(finite, vals, 0.0).astype(np.float64, copy=False)
        cnt = np.cumsum(finite.astype(np.float64), axis=0)
        s1 = np.cumsum(clean, axis=0)
        s2 = np.cumsum(clean * clean, axis=0)
        cnt = np.vstack([np.zeros((1, len(cols))), cnt])
        s1 = np.vstack([np.zeros((1, len(cols))), s1])
        s2 = np.vstack([np.zeros((1, len(cols))), s2])
        end = np.arange(1, len(pos) + 1)
        start = np.maximum(0, end - window)
        nobs = cnt[end] - cnt[start]
        sums = s1[end] - s1[start]
        sumsq = s2[end] - s2[start]
        mean = sums / np.maximum(nobs, 1.0)
        var = (sumsq - sums * sums / np.maximum(nobs, 1.0)) / np.maximum(nobs - 1.0, 1.0)
        z = (clean - mean) / (np.sqrt(np.maximum(var, 0.0)) + 1e-8)
        z[(nobs < minp) | ~finite] = np.nan
        out[pos] = z.astype(np.float32)
        del vals, finite, clean, cnt, s1, s2, nobs, sums, sumsq, mean, var, z
    return pd.DataFrame(out, index=panel.index, columns=cols)


VIEWS = {"raw": None, "tsz": _tsz, "csz": _csz, "csr": _csr}


def build():
    cfg = load_config(); np.random.seed(cfg["seed"])
    out_dir = cfg["output_dir"]; rep = os.path.join(cfg["reports_dir"], "plan_a")
    os.makedirs(rep, exist_ok=True)
    init_journal("plan_a", "Plan A · 大因子库(内存安全) 实时日志")
    os.makedirs(BIG_DIR, exist_ok=True)
    big = os.path.join(BIG_DIR, "data_factors_big.parquet")
    panel_ckpt = os.path.join(BIG_DIR, f"panel_ts_v2_{FACTOR_START}.parquet")
    sel_file = os.path.join(out_dir, "selected_factors.txt")
    if os.path.exists(big) and os.path.exists(sel_file):
        print(f"[build] exists: {big}", flush=True); return

    log_attempt("plan_a", "构建大因子库(内存安全重写)",
                f"float32+行过滤>={FACTOR_START}+视图逐个采样去重, 规避100GB OOM")
    t0 = time.time()
    data = load_all(cfg, start_date=FACTOR_START, end_date=cfg["end_date"])
    data = add_sessions_all(data); data = build_labels_all(data)
    print(f"raw+sess+labels: {len(data)} rows {time.time()-t0:.0f}s", flush=True)

    fstart = pd.Timestamp(FACTOR_START)
    if os.path.exists(panel_ckpt):
        t0 = time.time()
        panel = pd.read_parquet(panel_ckpt)
        print(f"loaded TS checkpoint: {panel.shape} {time.time()-t0:.0f}s", flush=True)
    else:
        # --- TS factors per symbol -> float32. CRITICAL: filter rows to FACTOR_START
        #     INSIDE the loop (factors computed on full history, but only 2017+ rows kept)
        #     so we never hold all 11.5M rows x all factors -> avoids OOM. ---
        t0 = time.time(); parts = []
        for i, (sym, g) in enumerate(data.groupby("symbol")):
            ff = compute_symbol_factors(g)            # full history (rolling needs warmup)
            ff = ff[ff["datetime"] >= fstart]          # keep only 2017+ rows
            for c in ff.columns:
                if c not in ("symbol", "datetime", "label"):
                    ff[c] = ff[c].astype(np.float32)
            ff["symbol"] = ff["symbol"].astype("category")
            parts.append(ff)
            if (i + 1) % 10 == 0:
                print(f"  factors {i+1} symbols, parts rows so far~{sum(len(p) for p in parts)}", flush=True)
        panel = pd.concat(parts, ignore_index=True)
        del parts; gc.collect()
        # Parts are produced from data.groupby("symbol") after load_all sorted by
        # symbol/datetime, so concat preserves the required order. A full sort over
        # 14M x 500 columns is very expensive and unnecessary here.
        panel["symbol"] = panel["symbol"].astype(str)
        panel.to_parquet(panel_ckpt, index=False)
        print(f"saved TS checkpoint: {panel_ckpt}", flush=True)
    ts_cols = [c for c in panel.columns if c not in ("symbol", "datetime", "label")]
    print(f"TS factors: {len(ts_cols)} cols, {len(panel)} rows (>={FACTOR_START}) {time.time()-t0:.0f}s", flush=True)
    log_result("plan_a", "TS因子", {"n_ts": len(ts_cols), "rows_kept": len(panel)})

    # --- sample rows (with label) for dedup ---
    lab_idx = panel.index[panel["label"].notna()].to_numpy()
    samp_idx = np.sort(np.random.RandomState(42).choice(lab_idx, min(SAMPLE_N, len(lab_idx)), replace=False))
    y = panel.loc[samp_idx, "label"].values.astype(np.float64)

    # --- build candidate sample matrix view-by-view (bounded memory) ---
    t0 = time.time()
    cand_names, cand_cols_sample, ic_abs = [], [], {}
    raw_ranked_cols = None
    for vname, fn in VIEWS.items():
        view_cols = ts_cols
        if vname == "tsz" and raw_ranked_cols is not None:
            view_cols = raw_ranked_cols[:300]
        elif vname in ("csz", "csr") and raw_ranked_cols is not None:
            view_cols = raw_ranked_cols[:900]
        if vname == "csr":
            sample_dates = pd.Index(panel.loc[samp_idx, "datetime"].unique())
            mask = panel["datetime"].isin(sample_dates)
            view = _csr(panel.loc[mask, ["datetime"] + view_cols], view_cols)
        elif fn is None:
            view = panel[view_cols]
        else:
            view = fn(panel, view_cols)
        sub = view.loc[samp_idx]
        for c in view_cols:
            name = c if vname == "raw" else f"{vname}_{c}"
            col = sub[c].values.astype(np.float64)
            col = np.nan_to_num(col, posinf=0, neginf=0)
            cov = np.mean(np.abs(col) > 0)
            if cov < 0.3:
                continue
            cand_names.append(name)
            cand_cols_sample.append(col.astype(np.float32))
            ic_abs[name] = abs(compute_ic(col, y))
        del view, sub; gc.collect()
        if vname == "raw":
            raw_ranked_cols = sorted(
                [n for n in cand_names if not any(n.startswith(p + "_") for p in ("tsz", "csz", "csr"))],
                key=lambda n: ic_abs[n],
                reverse=True,
            )
        print(f"  view {vname}: cum candidates={len(cand_names)} {time.time()-t0:.0f}s", flush=True)

    S = np.array(cand_cols_sample, dtype=np.float32).T   # (n_samp, n_cand)
    del cand_cols_sample; gc.collect()
    log_result("plan_a", "候选生成", {"n_candidates": len(cand_names)})

    # --- greedy correlation dedup (|corr|<0.9), priority by |IC| ---
    t0 = time.time()
    order = sorted(range(len(cand_names)), key=lambda j: ic_abs[cand_names[j]], reverse=True)
    Sn = (S - S.mean(0)) / (S.std(0) + 1e-9)
    selected_j, sel_mat = [], None
    for j in order:
        if ic_abs[cand_names[j]] < 1e-4:
            continue
        v = Sn[:, j]
        if sel_mat is None:
            selected_j.append(j); sel_mat = v.reshape(-1, 1); continue
        corr = np.abs(sel_mat.T @ v) / (len(v))
        if corr.max() < CORR_THRESH:
            selected_j.append(j)
            sel_mat = np.concatenate([sel_mat, v.reshape(-1, 1)], axis=1)
    selected = [cand_names[j] for j in selected_j]
    print(f"dedup: {len(selected)} factors (corr<{CORR_THRESH}) {time.time()-t0:.0f}s", flush=True)
    log_result("plan_a", "相关去重", {"n_selected": len(selected)},
               analysis="达标>1000" if len(selected) >= TARGET_MIN else f"仅{len(selected)},需扩因子")
    with open(sel_file, "w") as fh:
        fh.write("\n".join(selected))

    # --- single-factor IC screen ---
    ic_rank = pd.Series({n: ic_abs[n] for n in selected}).sort_values(ascending=False)
    ic_rank.to_csv(os.path.join(rep, "single_factor_ic.csv"))
    print("Top single-factor |IC|:\n" + ic_rank.head(20).to_string(), flush=True)
    log_shared_conclusion("大因子库完成",
        f"{len(selected)}因子(两两<0.9). Top: " + ", ".join(f"{k}={v:.3f}" for k, v in ic_rank.head(8).items()))

    # --- materialize selected in month chunks (avoid holding 14M x 1000 final) ---
    t0 = time.time()
    by_view = {}
    for name in selected:
        for v in ("tsz", "csz", "csr"):
            if name.startswith(v + "_"):
                by_view.setdefault(v, []).append(name[len(v) + 1:]); break
        else:
            by_view.setdefault("raw", []).append(name)
    tmp_big = big + ".tmp"
    if os.path.exists(tmp_big):
        os.remove(tmp_big)

    # Meta needed by the transformer's window builder.
    meta_cols = [c for c in ("symbol", "datetime", "is_long_break_before", "session_id",
                             "close", "open", "high", "low", "volume") if c in data.columns]
    panel_month = panel["datetime"].values.astype("datetime64[M]")
    data_month = data["datetime"].values.astype("datetime64[M]")
    months = np.unique(panel_month)
    sym_indices = panel.groupby("symbol", sort=False).indices
    writer = None
    rows_written = 0
    for mi, month in enumerate(months, 1):
        idx_month = np.where(panel_month == month)[0]
        if len(idx_month) == 0:
            continue
        base_cols = ["symbol", "datetime", "label"] + by_view.get("raw", [])
        chunk = panel.iloc[idx_month][base_cols].copy().reset_index(drop=True)

        if by_view.get("tsz"):
            warm_parts = []
            for _, idx in sym_indices.items():
                idx = np.asarray(idx, dtype=np.int64)
                hit = idx[panel_month[idx] == month]
                if len(hit) == 0:
                    continue
                # Include 119 preceding rows in the same symbol for rolling z warmup.
                loc0 = max(0, np.searchsorted(idx, hit[0]) - 119)
                loc1 = np.searchsorted(idx, hit[-1])
                warm_parts.append(idx[loc0:loc1 + 1])
            warm_idx = np.unique(np.concatenate(warm_parts)) if warm_parts else idx_month
            ts_panel = panel.iloc[warm_idx][["symbol", "datetime"] + by_view["tsz"]]
            view = _tsz(ts_panel, by_view["tsz"])
            view = view.loc[panel.index[idx_month]]
            view.columns = [f"tsz_{c}" for c in by_view["tsz"]]
            chunk = pd.concat([chunk, view.reset_index(drop=True)], axis=1)
            del ts_panel, view, warm_idx, warm_parts; gc.collect()

        for v in ("csz", "csr"):
            if by_view.get(v):
                sub_panel = panel.iloc[idx_month][["datetime"] + by_view[v]]
                view = VIEWS[v](sub_panel, by_view[v])
                view.columns = [f"{v}_{c}" for c in by_view[v]]
                chunk = pd.concat([chunk, view.reset_index(drop=True)], axis=1)
                del sub_panel, view; gc.collect()

        meta_df = data.loc[data_month == month, meta_cols]
        chunk = chunk.merge(meta_df, on=["symbol", "datetime"], how="left")
        for c in chunk.columns:
            if c not in ("symbol", "datetime"):
                if pd.api.types.is_float_dtype(chunk[c]):
                    chunk[c] = chunk[c].astype(np.float32)
        table = pa.Table.from_pandas(chunk, preserve_index=False)
        if writer is None:
            writer = pq.ParquetWriter(tmp_big, table.schema, compression="snappy")
        writer.write_table(table)
        rows_written += len(chunk)
        print(f"  materialized {mi}/{len(months)} {str(month)[:7]} rows={len(chunk)} total={rows_written}", flush=True)
        del chunk, meta_df, table; gc.collect()
    if writer is not None:
        writer.close()
    os.replace(tmp_big, big)
    print(f"saved {big}: rows={rows_written}, factors={len(selected)} {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    build()
