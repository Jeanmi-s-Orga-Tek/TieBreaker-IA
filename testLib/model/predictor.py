"""Prediction helpers for the persisted DecisionTree model."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from ..features import feature_vector_from_matches
from ..io.serializer import load_model, load_metadata


MODEL_PATH = Path(__file__).resolve().parents[1] / "io" / "model.pkl"


def _to_frame(input_data: Iterable[dict] | dict | Sequence[Sequence[float]]) -> pd.DataFrame:
    """Normalize user input into a DataFrame."""
    if isinstance(input_data, dict):
        return pd.DataFrame([input_data])
    if isinstance(input_data, Iterable):
        first = next(iter(input_data))
        if isinstance(first, dict):
            return pd.DataFrame(list(input_data))
        arr = np.asarray(list(input_data))
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        return pd.DataFrame(arr)
    raise TypeError("Unsupported input type for prediction")


def predict(input_data):
    """Load the model from disk and run predictions."""
    model = load_model(MODEL_PATH)
    X = _to_frame(input_data)
    return model.predict(X)


def predict_match(match_a: pd.Series, match_b: pd.Series, player_a: str, player_b: str) -> dict:
    """Predict winner between two players and surface model metadata."""
    features = feature_vector_from_matches(match_a, match_b, player_a, player_b)
    frame = pd.DataFrame([features]).drop(columns=["p1_name", "p2_name"], errors="ignore")

    model = load_model(MODEL_PATH)
    raw_pred = model.predict(frame)[0]
    proba = None
    if hasattr(model, "predict_proba"):
        proba = float(model.predict_proba(frame)[0][1])

    if raw_pred == 1:
        winner, loser = player_a, player_b
        confidence = proba
    else:
        winner, loser = player_b, player_a
        confidence = 1 - proba if proba is not None else None

    metadata = load_metadata() or {}
    return {
        "winner": winner,
        "loser": loser,
        "confidence": confidence,
        "model": {
            "accuracy": metadata.get("accuracy"),
            "train_accuracy": metadata.get("train_accuracy"),
            "tree_depth": metadata.get("tree_depth"),
            "tree_leaves": metadata.get("tree_leaves"),
            "saved_at": metadata.get("saved_at"),
        },
        "players": {"p1": player_a, "p2": player_b},
        "features": features,
    }
