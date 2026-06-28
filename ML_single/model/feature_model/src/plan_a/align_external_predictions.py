"""
Align external prediction files to an audited base prediction grid.

The output files keep the base `symbol, datetime, label` grid and attach one
external `pred` column, filling missing/non-finite predictions with zero.  This
is useful before feeding external experts into leak-free ensemble scripts that
inner-join component files.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, "/root/feature_model")

import numpy as np
import pandas as pd

from src.evaluation.metrics import compute_ic, ic_by_period


def parse_source(raw: str) -> tuple[str, Path]:
    if "=" not in raw:
        raise argparse.ArgumentTypeError("sources must be name=/path/file.parquet")
    name, path = raw.split("=", 1)
    name = name.strip()
    if not name:
        raise argparse.ArgumentTypeError("source name is empty")
    return name, Path(path)


def series_at(series: pd.Series, key: str) -> float:
    out = series.copy()
    out.index = out.index.map(str)
    return float(out.loc[str(key)])


def audit_frame(df: pd.DataFrame, name: str, file_name: str, missing_filled: int) -> dict:
    monthly = ic_by_period(df, "pred", "label", "M")
    yearly = ic_by_period(df, "pred", "label", "Y")
    return {
        "name": name,
        "file": file_name,
        "rows": len(df),
        "missing_filled": missing_filled,
        "ic": float(compute_ic(df["pred"].to_numpy(), df["label"].to_numpy())),
        "mmean": float(monthly.mean()),
        "mstd": float(monthly.std()),
        "ir": float(monthly.mean() / monthly.std()),
        "ic2018": series_at(yearly, "2018"),
        "ic2019": series_at(yearly, "2019"),
        "ic2020": series_at(yearly, "2020"),
        "ic2021": series_at(yearly, "2021"),
        "m202003": series_at(monthly, "2020-03"),
        "m202004": series_at(monthly, "2020-04"),
        "m202102": series_at(monthly, "2021-02"),
    }


def align_one(base: pd.DataFrame, name: str, src_path: Path, output_dir: Path) -> dict:
    src = pd.read_parquet(src_path, columns=["symbol", "datetime", "pred"])
    src["datetime"] = pd.to_datetime(src["datetime"])
    src = src.rename(columns={"pred": "_external_pred"})
    merged = base.merge(src, on=["symbol", "datetime"], how="left", validate="one_to_one")
    missing = int(merged["_external_pred"].isna().sum())
    pred = np.nan_to_num(
        merged["_external_pred"].astype("float64").to_numpy(),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    out = merged[["symbol", "datetime", "label"]].copy()
    out["pred"] = pred.astype("float64", copy=False)
    out_path = output_dir / f"predictions_{name}_aligned.parquet"
    out.to_parquet(out_path, index=False)
    return audit_frame(out, name=name, file_name=out_path.name, missing_filled=missing)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True, type=Path)
    parser.add_argument("--source", action="append", type=parse_source, required=True)
    parser.add_argument("--output-dir", default=Path("outputs"), type=Path)
    parser.add_argument("--audit", default=Path("reports/plan_a/e2e_aligned_external_audit.csv"), type=Path)
    parser.add_argument("--start-date", default="2018-01-01")
    args = parser.parse_args()

    base = pd.read_parquet(args.base, columns=["symbol", "datetime", "label"])
    base["datetime"] = pd.to_datetime(base["datetime"])
    base = base[base["datetime"] >= pd.Timestamp(args.start_date)].copy()
    base = base.sort_values(["symbol", "datetime"]).reset_index(drop=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.audit.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for name, src_path in args.source:
        rows.append(align_one(base, name, src_path, args.output_dir))

    audit = pd.DataFrame(rows)
    audit.to_csv(args.audit, index=False)
    print(audit.to_string(index=False))


if __name__ == "__main__":
    main()
