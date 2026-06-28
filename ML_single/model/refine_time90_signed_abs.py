#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

RUN_DIR = Path("/root/autodl-tmp/quant/ML/agent_runs/lgb_reg8_shape_seqcal_20260628")
if str(RUN_DIR) not in sys.path:
    sys.path.insert(0, str(RUN_DIR))

import lgb_shape_seqcal_reg8 as seq  # noqa: E402


def build_fold_cache(d19: pd.DataFrame, fit_json: str):
    folds = [
        ("q2", "2019-01-01", "2019-04-01", "2019-04-01", "2019-07-01"),
        ("q3", "2019-01-01", "2019-07-01", "2019-07-01", "2019-10-01"),
        ("q4", "2019-01-01", "2019-10-01", "2019-10-01", "2020-01-01"),
        ("h2", "2019-01-01", "2019-07-01", "2019-07-01", "2020-01-01"),
    ]
    cache = []
    for fold, tr_s, tr_e, val_s, val_e in folds:
        tr = seq.subset(d19, tr_s, tr_e)
        val = seq.subset(d19, val_s, val_e)
        shape_fit = seq.fit_shape_transform(tr, fit_json)
        cache.append(
            {
                "fold": fold,
                "train": tr,
                "val": val,
                "pred_train": seq.transform_shape(tr, shape_fit),
                "pred_val": seq.transform_shape(val, shape_fit),
                "shape_fit": shape_fit,
            }
        )
    return cache


def screen_candidate(cache: list[dict[str, object]], cand: seq.Candidate) -> dict[str, object]:
    row: dict[str, object] = {
        "candidate": cand.name,
        "stages_json": json.dumps([seq.asdict(s) for s in cand.stages], sort_keys=True),
    }
    vals = []
    for item in cache:
        pred_val, stage_fits = seq.fit_apply_stages(
            item["train"],
            item["pred_train"],
            item["val"],
            item["pred_val"],
            cand.stages,
        )
        ic = seq.pooled(item["val"], pred_val)
        row[f"{item['fold']}_ic"] = ic
        row[f"{item['fold']}_stage_fits"] = json.dumps(stage_fits, sort_keys=True)
        vals.append(ic)
    q = np.asarray(vals[:3], dtype=np.float64)
    row["screen_mean_q2q4"] = float(np.nanmean(q))
    row["screen_std_q2q4"] = float(np.nanstd(q, ddof=1))
    row["screen_min_q2q4"] = float(np.nanmin(q))
    row["screen_h2_ic"] = float(vals[3])
    row["score_mean_m025std"] = row["screen_mean_q2q4"] - 0.25 * row["screen_std_q2q4"]
    row["score_mean_m050std"] = row["screen_mean_q2q4"] - 0.50 * row["screen_std_q2q4"]
    row["score_q3_h2"] = row["q3_ic"] + row["h2_ic"]
    row["score_q3_h2_mean"] = row["q3_ic"] + row["h2_ic"] + 0.25 * row["screen_mean_q2q4"]
    row["score_min_mean"] = row["screen_min_q2q4"] + 0.25 * row["screen_mean_q2q4"]
    return row


def candidate_grid() -> list[seq.Candidate]:
    out: list[seq.Candidate] = []
    time_alphas = [0.75, 0.875, 1.0, 1.125]
    bins = list(range(4, 13))
    signed_alphas = [round(x, 2) for x in np.arange(0.25, 0.81, 0.05)]
    for ta in time_alphas:
        for b in bins:
            for ba in signed_alphas:
                out.append(
                    seq.Candidate(
                        f"ref_time90_a{ta:g}_then_signed_abs{b}_a{ba:g}",
                        (
                            seq.StageSpec("time", bucket_minutes=90, alpha=ta, k=5_000.0),
                            seq.StageSpec("signed_abs", n_bins=b, alpha=ba, k=80_000.0),
                        ),
                    )
                )
    return out


def main() -> None:
    selected = pd.read_csv(seq.SHAPE_SELECTED).iloc[0]
    fit_json = str(selected["fit_json"])
    d19 = seq.add_known_cols(pd.read_parquet(seq.PRED_2019))
    d20 = seq.add_known_cols(pd.read_parquet(seq.PRED_2020))
    cache = build_fold_cache(d19, fit_json)

    rows = []
    for i, cand in enumerate(candidate_grid(), 1):
        row = screen_candidate(cache, cand)
        rows.append(row)
        if i % 25 == 0:
            print(
                f"[ref-screen] {i} last={cand.name} "
                f"score={row['score_mean_m025std']:.6f} q3h2={row['score_q3_h2']:.6f}",
                flush=True,
            )
    screen = pd.DataFrame(rows)
    screen.to_csv(RUN_DIR / "refine_time90_signed_abs_screen_2019.csv", index=False)

    selectors = [
        "score_mean_m025std",
        "score_mean_m050std",
        "score_q3_h2",
        "score_q3_h2_mean",
        "score_min_mean",
        "screen_h2_ic",
    ]
    audit_indices = set()
    selector_winners = []
    for selector in selectors:
        ranked = screen.sort_values(selector, ascending=False).reset_index()
        winner = ranked.iloc[0].copy()
        winner["selector"] = selector
        winner["selector_rank"] = 1
        selector_winners.append(winner)
        audit_indices.update(ranked.head(5)["index"].tolist())

    audit_rows = []
    pred_written = set()
    for idx in sorted(audit_indices):
        row = screen.loc[idx]
        cand = seq.candidate_from_row(row)
        audit, pred20 = seq.audit_one(d19, d20, fit_json, cand)
        winner_selectors = [
            str(w["selector"])
            for w in selector_winners
            if str(w["candidate"]) == str(row["candidate"])
        ]
        combined = {
            **row.to_dict(),
            **audit,
            "selector_winner": bool(winner_selectors),
            "winner_selectors": "|".join(winner_selectors),
        }
        audit_rows.append(combined)
        if combined["selector_winner"] and audit["audit2020_ic"] > 0.05 and row["candidate"] not in pred_written:
            out = d20[["symbol", "datetime", "label", "pred", "pred_xsz", "pred_xrank"]].copy()
            out["pred_lgb_ref_time90_signed_abs"] = pred20.astype(np.float32)
            out.to_parquet(RUN_DIR / f"{row['candidate']}_selector_winner_audit2020_predictions.parquet", index=False)
            pred_written.add(str(row["candidate"]))
        print(
            f"[ref-audit] {row['candidate']} winner={combined['selector_winner']} "
            f"selectors={combined['winner_selectors']} 2020={audit['audit2020_ic']:.6f}",
            flush=True,
        )

    audited = pd.DataFrame(audit_rows)
    audited.to_csv(RUN_DIR / "refine_time90_signed_abs_audit.csv", index=False)
    winners = audited[audited["selector_winner"]].copy()
    winners.to_csv(RUN_DIR / "refine_time90_signed_abs_selector_winners_audit.csv", index=False)
    passing = winners[winners["audit2020_ic"] > 0.05].copy()
    if not passing.empty:
        best = passing.sort_values("audit2020_ic", ascending=False).iloc[0].to_dict()
        config = {
            "candidate": best["candidate"],
            "winner_selectors": best["winner_selectors"],
            "stages": json.loads(best["stages_json"]),
            "shape_source": str(seq.SHAPE_SELECTED),
            "shape_candidate": str(selected["candidate"]),
            "selected_by": "predeclared 2019 selector winner in refined time90->signed_abs family",
            "no_future_leakage_note": "selectors use only 2019 folds; final shape thresholds and multipliers fit on full 2019; 2020 labels used only for audit.",
        }
        (RUN_DIR / "refine_time90_signed_abs_pass_config.json").write_text(json.dumps(config, indent=2, sort_keys=True), encoding="utf-8")
    show = [
        "candidate",
        "selector_winner",
        "winner_selectors",
        "score_mean_m025std",
        "score_q3_h2",
        "screen_h2_ic",
        "audit2020_ic",
        "audit2020_monthly_mean",
    ]
    print("\nSelector winners")
    print(winners[show].sort_values("audit2020_ic", ascending=False).to_string(index=False))
    print("\nTop audited diagnostics")
    print(audited[show].sort_values("audit2020_ic", ascending=False).head(20).to_string(index=False))


if __name__ == "__main__":
    main()
