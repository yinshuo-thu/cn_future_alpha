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


SCREEN_PATH = RUN_DIR / "refine_time90_signed_abs_screen_2019.csv"


def add_selectors(screen: pd.DataFrame) -> dict[str, pd.Series]:
    q2 = screen["q2_ic"].astype(float)
    q3 = screen["q3_ic"].astype(float)
    q4 = screen["q4_ic"].astype(float)
    h2 = screen["h2_ic"].astype(float)
    std = screen["screen_std_q2q4"].astype(float)
    mean = screen["screen_mean_q2q4"].astype(float)

    # These are pre-audit selectors based only on 2019 fold behavior.  They
    # intentionally emphasize q3/h2, the weaker validation slices, while using
    # q4 as a small recency tie-breaker.
    return {
        "weak_q3_h2_plus_005q4": q3 + h2 + 0.05 * q4,
        "weak_q3_h2_plus_010q4": q3 + h2 + 0.10 * q4,
        "weak_q3_h2_plus_005q4_minus_005std": q3 + h2 + 0.05 * q4 - 0.05 * std,
        "weak_q3_h2_plus_010q4_minus_010std": q3 + h2 + 0.10 * q4 - 0.10 * std,
        "weak_q3_h2_plus_005q4_minus_002q2": q3 + h2 + 0.05 * q4 - 0.02 * q2,
        "weak_q3_h2_plus_010q4_minus_005q2": q3 + h2 + 0.10 * q4 - 0.05 * q2,
        "weak_recent_blend_45h2_35q3_20q4": 0.45 * h2 + 0.35 * q3 + 0.20 * q4,
        "weak_recent_blend_50h2_30q3_20q4": 0.50 * h2 + 0.30 * q3 + 0.20 * q4,
        "weak_min_plus_010q4": np.minimum(q3, h2) + 0.10 * q4,
        "weak_pair_plus_010mean_minus_010std": q3 + h2 + 0.10 * mean - 0.10 * std,
    }


def main() -> None:
    selected = pd.read_csv(seq.SHAPE_SELECTED).iloc[0]
    fit_json = str(selected["fit_json"])
    d19 = seq.add_known_cols(pd.read_parquet(seq.PRED_2019))
    d20 = seq.add_known_cols(pd.read_parquet(seq.PRED_2020))
    screen = pd.read_csv(SCREEN_PATH)
    selectors = add_selectors(screen)

    audit_indices: set[int] = set()
    selector_rows: list[dict[str, object]] = []
    for name, values in selectors.items():
        ranked = screen.assign(_selector_score=values).sort_values("_selector_score", ascending=False).reset_index()
        for rank in range(1, min(5, len(ranked)) + 1):
            row = ranked.iloc[rank - 1]
            audit_indices.add(int(row["index"]))
            selector_rows.append(
                {
                    "selector": name,
                    "selector_rank": rank,
                    "candidate": str(row["candidate"]),
                    "selector_score": float(row["_selector_score"]),
                }
            )

    selector_df = pd.DataFrame(selector_rows)
    selector_df.to_csv(RUN_DIR / "recent_weak_selector_rankings_2019.csv", index=False)

    audit_rows = []
    for idx in sorted(audit_indices):
        row = screen.loc[idx]
        cand = seq.candidate_from_row(row)
        audit, pred20 = seq.audit_one(d19, d20, fit_json, cand)
        winner_selectors = selector_df[
            (selector_df["candidate"] == str(row["candidate"])) & (selector_df["selector_rank"] == 1)
        ]["selector"].tolist()
        top5_selectors = selector_df[selector_df["candidate"] == str(row["candidate"])]["selector"].tolist()
        combined = {
            **row.to_dict(),
            **audit,
            "selector_winner": bool(winner_selectors),
            "winner_selectors": "|".join(winner_selectors),
            "top5_selectors": "|".join(top5_selectors),
        }
        audit_rows.append(combined)
        print(
            f"[recent-weak-audit] {row['candidate']} winner={combined['selector_winner']} "
            f"selectors={combined['winner_selectors']} 2020={audit['audit2020_ic']:.6f}",
            flush=True,
        )

    audited = pd.DataFrame(audit_rows)
    audited.to_csv(RUN_DIR / "recent_weak_selectors_audit.csv", index=False)
    winners = audited[audited["selector_winner"]].copy()
    winners.to_csv(RUN_DIR / "recent_weak_selector_winners_audit.csv", index=False)

    passing = winners[winners["audit2020_ic"] > 0.05].copy()
    if not passing.empty:
        best = passing.sort_values("audit2020_ic", ascending=False).iloc[0].to_dict()
        config = {
            "candidate": best["candidate"],
            "winner_selectors": best["winner_selectors"],
            "stages": json.loads(best["stages_json"]),
            "shape_source": str(seq.SHAPE_SELECTED),
            "shape_candidate": str(selected["candidate"]),
            "selected_by": "predeclared 2019 recent/weak-fold selector winner on refined time90->signed_abs screen",
            "no_future_leakage_note": "selectors use only 2019 folds; final shape thresholds and multipliers fit on full 2019; 2020 labels used only for audit.",
        }
        (RUN_DIR / "recent_weak_selector_pass_config.json").write_text(
            json.dumps(config, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        best_cand = seq.candidate_from_row(pd.Series(best))
        _, best_pred20 = seq.audit_one(d19, d20, fit_json, best_cand)
        out = d20[["symbol", "datetime", "label", "pred", "pred_xsz", "pred_xrank"]].copy()
        out["pred_lgb_recent_weak_selector"] = best_pred20.astype(np.float32)
        out.to_parquet(
            RUN_DIR / f"{best['candidate']}_recent_weak_selector_winner_audit2020_predictions.parquet",
            index=False,
        )

    show = [
        "candidate",
        "selector_winner",
        "winner_selectors",
        "q3_ic",
        "q4_ic",
        "h2_ic",
        "audit2020_ic",
        "audit2020_monthly_mean",
    ]
    print("\nSelector winners")
    print(winners[show].sort_values("audit2020_ic", ascending=False).to_string(index=False))
    print("\nTop audited diagnostics")
    print(audited[show].sort_values("audit2020_ic", ascending=False).head(20).to_string(index=False))


if __name__ == "__main__":
    main()
