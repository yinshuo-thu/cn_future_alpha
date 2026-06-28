import os
import json

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


DEFAULT_FACTOR_DATA = "/root/shared-nvme/feature_model/data_factors_big.parquet"
META_COLS = {
    "symbol",
    "datetime",
    "label",
    "session_id",
    "is_long_break_before",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "amount",
    "oi",
}


def factor_data_path():
    return os.environ.get("FACTOR_DATA_PATH", DEFAULT_FACTOR_DATA)


def select_factor_columns(cfg, data_path, top_n, selected_name="selected_factors.txt"):
    names = pq.ParquetFile(data_path).schema.names
    schema = set(names)
    mode = os.environ.get("FACTOR_SELECTION", "selected_then_schema")
    accepted_path = os.environ.get(
        "ACCEPTED_FACTORS_PATH",
        "/root/feature_model/factor_mining/store/accepted.jsonl",
    )
    if mode in {"accepted_abs_ic", "accepted_order"} and os.path.exists(accepted_path):
        accepted = []
        with open(accepted_path) as f:
            for line in f:
                row = json.loads(line)
                if row.get("name") in schema and row.get("name") not in META_COLS:
                    accepted.append(row)
        if mode == "accepted_abs_ic":
            accepted.sort(key=lambda r: abs(float(r.get("ic", 0.0))), reverse=True)
        feat_cols = []
        seen = set()
        for row in accepted:
            col = row["name"]
            if col not in seen:
                feat_cols.append(col)
                seen.add(col)
                if len(feat_cols) >= top_n:
                    return feat_cols
        for col in names:
            if col not in META_COLS and col not in seen:
                feat_cols.append(col)
                seen.add(col)
                if len(feat_cols) >= top_n:
                    break
        if feat_cols:
            return feat_cols

    selected_path = os.path.join(cfg["output_dir"], selected_name)
    selected = []
    if os.path.exists(selected_path):
        with open(selected_path) as f:
            selected = [x.strip() for x in f if x.strip()]

    feat_cols = []
    seen = set()
    for col in selected:
        if col in schema and col not in META_COLS and col not in seen:
            feat_cols.append(col)
            seen.add(col)
            if len(feat_cols) >= top_n:
                return feat_cols

    for col in names:
        if col not in META_COLS and col not in seen:
            feat_cols.append(col)
            seen.add(col)
            if len(feat_cols) >= top_n:
                break

    if not feat_cols:
        raise ValueError(f"no factor columns found in {data_path}")
    return feat_cols


def add_session_id_if_missing(data, gap_minutes=60):
    if "session_id" in data.columns:
        return data
    out = data.sort_values(["symbol", "datetime"]).copy()
    gap = out.groupby("symbol")["datetime"].diff().dt.total_seconds().div(60)
    new_session = gap.isna() | (gap > gap_minutes)
    session_num = new_session.groupby(out["symbol"]).cumsum().astype(np.int32)
    sym_code = pd.factorize(out["symbol"], sort=True)[0].astype(np.int32)
    out["session_id"] = (sym_code.astype(np.int64) << 32) + session_num.to_numpy(np.int64)
    return out
