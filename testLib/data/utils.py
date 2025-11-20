"""Data loading helpers for the DecisionTree pipeline."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Tuple

import pandas as pd

DEFAULT_DATA_PATH = Path(__file__).resolve().parent / "data.csv"


def dataset_exists(path: Path | None = None) -> bool:
    """Return True if a dataset file already exists."""
    data_path = Path(path) if path else DEFAULT_DATA_PATH
    return data_path.is_file()


def load_dataset(path: Path | None = None) -> Tuple[pd.DataFrame, pd.Series]:
    """Load dataset from CSV and return features/labels."""
    data_path = Path(path) if path else DEFAULT_DATA_PATH
    if not data_path.is_file():
        raise FileNotFoundError(f"Dataset missing: {data_path}")

    df = pd.read_csv(data_path)
    if "target" not in df.columns:
        raise ValueError("Dataset must contain a 'target' column")

    X = df.drop(columns=["target"])
    y = df["target"]
    return X, y


def save_dataset(records: Iterable[dict], path: Path | None = None) -> Path:
    """Persist records (list of dicts) as CSV for reproducible training."""
    data_path = Path(path) if path else DEFAULT_DATA_PATH
    df = pd.DataFrame.from_records(records)
    if "target" not in df.columns:
        raise ValueError("Saved dataset must include 'target'")
    df.to_csv(data_path, index=False)
    return data_path
