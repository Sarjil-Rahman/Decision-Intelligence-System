from __future__ import annotations

import json
import hashlib
from pathlib import Path
from typing import Any, Iterable, List

import numpy as np
import pandas as pd

import os
import logging


def get_logger(name: str = "pipeline") -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)

    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(h)
    logger.propagate = False
    return logger


def require_files(data_dir: str, files: Iterable[str]) -> None:
    root = Path(data_dir).resolve()
    missing = []
    escaped = []
    for f in files:
        candidate = (root / f).resolve()
        try:
            candidate.relative_to(root)
        except ValueError:
            escaped.append(str(f))
            continue
        if not candidate.is_file():
            missing.append(str(f))
    if escaped:
        raise ValueError(
            f"Required file(s) in '{data_dir}' resolve outside the dataset: " + ", ".join(escaped)
        )
    if missing:
        raise FileNotFoundError(f"Missing required file(s) in '{data_dir}': " + ", ".join(missing))


def _json_default(o: Any):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    return str(o)


def write_json(path: str, obj: Any, indent: int = 2) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=indent, ensure_ascii=False, default=_json_default)
        f.write("\n")


def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def stable_config_hash(obj: Any) -> str:
    s = json.dumps(obj, sort_keys=True, ensure_ascii=False, default=_json_default)
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def select_representative_series_subset(
    df: pd.DataFrame,
    *,
    max_series: int | None,
    candidate_group_cols: List[str] | None = None,
    sort_col: str = "id",
) -> pd.DataFrame:
    """Select a deterministic, more representative subset than plain head().

    Why this exists:
    When a small dev subset uses the first N rows of M5, it usually over-represents
    a single category/store. That makes business summaries and dashboards look broader
    than they really are. This helper round-robins across available groups first.
    """
    if max_series is None or int(max_series) <= 0 or len(df) <= int(max_series):
        return df.copy()

    n = int(max_series)
    group_cols = [
        c
        for c in (candidate_group_cols or ["store_id", "cat_id", "state_id", "dept_id"])
        if c in df.columns
    ]
    work = df.copy()

    if sort_col in work.columns:
        work = work.sort_values(sort_col, kind="mergesort").reset_index(drop=True)
    else:
        work = work.reset_index(drop=True)

    if not group_cols:
        return work.iloc[:n].copy()

    work["_subset_group_key"] = work[group_cols].astype(str).agg(" | ".join, axis=1)
    work["_orig_index"] = work.index

    buckets = []
    for key, g in work.groupby("_subset_group_key", sort=True):
        g = g.reset_index(drop=True)
        buckets.append((key, g))

    picked_idx: List[int] = []
    cursor = 0
    while len(picked_idx) < n:
        made_progress = False
        for _, bucket in buckets:
            if cursor < len(bucket):
                picked_idx.append(int(bucket.loc[cursor, "_orig_index"]))
                made_progress = True
                if len(picked_idx) >= n:
                    break
        if not made_progress:
            break
        cursor += 1

    out = (
        work.iloc[picked_idx]
        .drop(columns=["_subset_group_key", "_orig_index"], errors="ignore")
        .copy()
    )
    return out.reset_index(drop=True)
