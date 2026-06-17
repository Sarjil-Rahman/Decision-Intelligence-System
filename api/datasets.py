from __future__ import annotations

import re
from pathlib import Path

REQUIRED_M5_FILES = (
    "sales_train_validation.csv",
    "calendar.csv",
    "sell_prices.csv",
    "sample_submission.csv",
)

DATASET_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")
SAFE_FILENAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")


def resolve_dataset_dir(data_root: str | Path, dataset_id: str) -> Path:
    if not DATASET_ID_RE.fullmatch(dataset_id or ""):
        raise ValueError("dataset_id must be a simple allow-listed identifier.")
    if any(sep in dataset_id for sep in ("/", "\\", "..")):
        raise ValueError("dataset_id must not contain path separators or traversal.")
    root = Path(data_root).resolve()
    candidate = (root / dataset_id).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise ValueError("dataset_id resolves outside DATA_ROOT.") from exc
    if not candidate.exists() or not candidate.is_dir():
        raise FileNotFoundError(f"Dataset not found: {dataset_id}")
    return candidate


def resolve_dataset_file(dataset_dir: str | Path, filename: str) -> Path:
    if not SAFE_FILENAME_RE.fullmatch(filename or ""):
        raise ValueError("file path must be a safe relative filename.")
    if any(sep in filename for sep in ("/", "\\", "..")) or Path(filename).is_absolute():
        raise ValueError("file path must not be absolute or contain traversal.")
    root = Path(dataset_dir).resolve()
    candidate = (root / filename).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise ValueError("file path resolves outside the authorised dataset.") from exc
    if candidate.exists() and not candidate.is_file():
        raise ValueError("file path must resolve to a regular file.")
    return candidate


def validate_required_dataset_files(dataset_dir: str | Path) -> None:
    root = Path(dataset_dir).resolve()
    missing_or_invalid: list[str] = []
    escaped: list[str] = []
    for name in REQUIRED_M5_FILES:
        candidate = (root / name).resolve()
        try:
            candidate.relative_to(root)
        except ValueError:
            escaped.append(name)
            continue
        if not candidate.is_file():
            missing_or_invalid.append(name)
    if escaped:
        raise ValueError(
            "Required dataset file resolves outside the authorised dataset: " + ", ".join(escaped)
        )
    if missing_or_invalid:
        raise FileNotFoundError("Missing required dataset files: " + ", ".join(missing_or_invalid))
