from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class RollingSplit:
    name: str
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    eval_start: pd.Timestamp
    eval_end: pd.Timestamp
    mode: str


def month_end_exclusive(year: int, month: int) -> pd.Timestamp:
    start = pd.Timestamp(year=year, month=month, day=1)
    return start + pd.offsets.MonthBegin(1)


def make_monthly_splits(
    eval_year: int,
    train_start: str = "2017-01-01",
    first_month: int = 1,
    last_month: int = 12,
    embargo_days: int = 1,
    mode: str = "rolling",
) -> list[RollingSplit]:
    splits = []
    for month in range(first_month, last_month + 1):
        eval_start = pd.Timestamp(year=eval_year, month=month, day=1)
        eval_end = month_end_exclusive(eval_year, month)
        train_end = eval_start - pd.Timedelta(days=int(embargo_days))
        splits.append(
            RollingSplit(
                name=f"{eval_year}-{month:02d}",
                train_start=pd.Timestamp(train_start),
                train_end=train_end,
                eval_start=eval_start,
                eval_end=eval_end,
                mode=mode,
            )
        )
    return splits


def make_representative_2019_splits(train_start: str = "2017-01-01") -> list[RollingSplit]:
    all_splits = make_monthly_splits(2019, train_start=train_start, mode="research_rep")
    keep = {"2019-01", "2019-04", "2019-07", "2019-10", "2019-12"}
    return [s for s in all_splits if s.name in keep]


def make_mode_c_smoke_split(train_start: str = "2017-01-01") -> RollingSplit:
    return make_monthly_splits(2019, train_start=train_start, first_month=1, last_month=1, mode="smoke")[0]
