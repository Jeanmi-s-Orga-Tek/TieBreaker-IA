"""Data subpackage exports and helpers."""
from .utils import load_dataset, save_dataset, dataset_exists
from .matches import (
    load_matches,
    latest_match_for_player,
    resolve_data_root,
)

__all__ = [
    "load_dataset",
    "save_dataset",
    "dataset_exists",
    "load_matches",
    "latest_match_for_player",
    "resolve_data_root",
]
