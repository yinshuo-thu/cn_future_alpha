from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Iterator

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, Sampler


@dataclass(frozen=True)
class SampleRef:
    array_id: int
    pos: int
    row_id: int


class WindowDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        feature_cols: list[str],
        target_cols: list[str],
        mask_cols: list[str],
        seq_len: int,
        row_indices: np.ndarray,
    ) -> None:
        self.feature_cols = feature_cols
        self.target_cols = target_cols
        self.mask_cols = mask_cols
        self.seq_len = int(seq_len)
        self.refs: list[SampleRef] = []
        self.arrays: list[dict[str, object]] = []
        row_set = set(int(i) for i in row_indices)
        for array_id, (symbol, grp) in enumerate(df.sort_values(["symbol", "datetime"]).groupby("symbol", sort=False)):
            pos_rows = grp.index.to_numpy(dtype=np.int64)
            features = grp[feature_cols].to_numpy(dtype=np.float32)
            targets = grp[target_cols].to_numpy(dtype=np.float32)
            masks = grp[mask_cols].to_numpy(dtype=bool)
            meta = grp[["datetime", "symbol", "sector", "session_type", "minute_of_day", "month", "session_pos"]].copy()
            self.arrays.append(
                {
                    "symbol": str(symbol),
                    "rows": pos_rows,
                    "features": features,
                    "targets": targets,
                    "masks": masks,
                    "meta": meta.reset_index(drop=True),
                }
            )
            for pos, row_id in enumerate(pos_rows):
                if int(row_id) not in row_set:
                    continue
                if pos + 1 < self.seq_len:
                    continue
                self.refs.append(SampleRef(array_id=array_id, pos=pos, row_id=int(row_id)))

    def __len__(self) -> int:
        return len(self.refs)

    def __getitem__(self, index: int) -> dict[str, object]:
        ref = self.refs[index]
        arr = self.arrays[ref.array_id]
        pos = ref.pos
        start = pos + 1 - self.seq_len
        features = arr["features"][start : pos + 1]
        targets = arr["targets"][pos]
        masks = arr["masks"][pos]
        meta = arr["meta"].iloc[pos]
        return {
            "x": torch.from_numpy(features),
            "targets": torch.from_numpy(targets),
            "masks": torch.from_numpy(masks.astype(np.float32)),
            "row_id": ref.row_id,
            "timestamp": meta["datetime"],
            "symbol": meta["symbol"],
            "sector": meta["sector"],
            "session_type": meta["session_type"],
            "minute_of_day": int(meta["minute_of_day"]),
            "month": int(meta["month"]),
        }


def collate_window_batch(batch: list[dict[str, object]]) -> dict[str, object]:
    return {
        "x": torch.stack([b["x"] for b in batch]),
        "targets": torch.stack([b["targets"] for b in batch]),
        "masks": torch.stack([b["masks"] for b in batch]),
        "row_id": np.asarray([b["row_id"] for b in batch], dtype=np.int64),
        "timestamp": [b["timestamp"] for b in batch],
        "symbol": [b["symbol"] for b in batch],
        "sector": [b["sector"] for b in batch],
        "session_type": [b["session_type"] for b in batch],
        "minute_of_day": torch.tensor([b["minute_of_day"] for b in batch], dtype=torch.long),
        "month": torch.tensor([b["month"] for b in batch], dtype=torch.long),
    }


class TimestampBatchSampler(Sampler[list[int]]):
    def __init__(
        self,
        dataset: WindowDataset,
        timestamps_per_batch: int = 16,
        symbols_per_timestamp: int = 32,
        min_symbols_per_timestamp: int = 10,
        shuffle: bool = True,
        seed: int = 11,
        max_batches: int | None = None,
    ) -> None:
        self.dataset = dataset
        self.timestamps_per_batch = int(timestamps_per_batch)
        self.symbols_per_timestamp = int(symbols_per_timestamp)
        self.min_symbols_per_timestamp = int(min_symbols_per_timestamp)
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self.max_batches = max_batches
        by_ts: dict[pd.Timestamp, list[int]] = defaultdict(list)
        for i, ref in enumerate(dataset.refs):
            arr = dataset.arrays[ref.array_id]
            ts = arr["meta"].iloc[ref.pos]["datetime"]
            by_ts[pd.Timestamp(ts)].append(i)
        self.timestamps = [ts for ts, idxs in by_ts.items() if len(idxs) >= self.min_symbols_per_timestamp]
        self.by_ts = {ts: by_ts[ts] for ts in self.timestamps}

    def __iter__(self) -> Iterator[list[int]]:
        rng = np.random.default_rng(self.seed)
        timestamps = list(self.timestamps)
        if self.shuffle:
            rng.shuffle(timestamps)
        batches = 0
        for start in range(0, len(timestamps), self.timestamps_per_batch):
            selected_ts = timestamps[start : start + self.timestamps_per_batch]
            batch: list[int] = []
            for ts in selected_ts:
                idxs = list(self.by_ts[ts])
                if self.shuffle and len(idxs) > self.symbols_per_timestamp:
                    idxs = rng.choice(idxs, size=self.symbols_per_timestamp, replace=False).tolist()
                else:
                    idxs = idxs[: self.symbols_per_timestamp]
                batch.extend(idxs)
            if batch:
                yield batch
                batches += 1
                if self.max_batches is not None and batches >= self.max_batches:
                    return

    def __len__(self) -> int:
        n = (len(self.timestamps) + self.timestamps_per_batch - 1) // self.timestamps_per_batch
        return min(n, int(self.max_batches)) if self.max_batches is not None else n


def valid_row_indices(
    df: pd.DataFrame,
    start: pd.Timestamp,
    end: pd.Timestamp,
    target_mask_col: str = "mask_proxy_ret_30m",
) -> np.ndarray:
    mask = (
        (df["datetime"] >= pd.Timestamp(start))
        & (df["datetime"] < pd.Timestamp(end))
        & df[target_mask_col].astype(bool)
    )
    return df.index[mask].to_numpy(dtype=np.int64)
