"""Training entry point for DecisionTree models."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

from ..features import feature_vector_from_matches
from ..data.matches import load_matches, latest_match_for_player
from ..io.serializer import save_model, save_metadata


def _unique_players(matches: pd.DataFrame) -> np.ndarray:
    combined = pd.concat([matches["winner_name"], matches["loser_name"]], ignore_index=True)
    return combined.dropna().astype(str).unique()


DATA_ROOT = Path(__file__).resolve().parents[2] / "data"


@dataclass
class TrainConfig:
    test_size: float = 0.2
    random_state: int = 42
    max_depth: Optional[int] = None
    years: Optional[int] = 6


def _build_training_samples(matches: pd.DataFrame) -> list[dict]:
    """Materialize feature dicts from the last available match of each player."""
    players = _unique_players(matches)
    records: list[dict] = []
    for player in players:
        last_match = latest_match_for_player(matches, player)
        if last_match is None:
            continue
        opponent = last_match["loser_name"] if last_match["winner_name"] == player else last_match["winner_name"]
        opp_match = latest_match_for_player(matches, opponent)
        if opp_match is None:
            continue
        features = feature_vector_from_matches(last_match, opp_match, player, opponent)
        features["target"] = 1 if last_match["winner_name"] == player else 0
        features["player"] = player
        features["opponent"] = opponent
        records.append(features)
    return records


def _summarize_training(
    cfg: TrainConfig,
    matches: pd.DataFrame,
    records: list[dict],
    clf: DecisionTreeClassifier,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    acc: float,
) -> dict:
    """Compile a detailed, human-friendly training report."""
    return {
        "accuracy": float(acc),
        "train_accuracy": float(clf.score(X_train, y_train)),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "n_features": int(X_train.shape[1]),
        "feature_names": X_train.columns.tolist(),
        "tree_depth": int(clf.get_depth()),
        "tree_leaves": int(clf.get_n_leaves()),
        "records_built": int(len(records)),
        "unique_players": int(len(_unique_players(matches))),
        "date_range": (
            matches["tourney_date"].dropna().min().date().isoformat()
            if matches["tourney_date"].notna().any()
            else None,
            matches["tourney_date"].dropna().max().date().isoformat()
            if matches["tourney_date"].notna().any()
            else None,
        ),
        "year_window": cfg.years,
    }


def train_model(cfg: TrainConfig | None = None, data_root: Path | None = None) -> dict:
    """Train model and store it on disk, returning a detailed metrics dict."""
    cfg = cfg or TrainConfig()
    root = data_root or DATA_ROOT

    matches = load_matches(root, limit_years=cfg.years or 6)
    records = _build_training_samples(matches)
    if not records:
        raise RuntimeError("No training samples could be generated from match history")

    df = pd.DataFrame(records).dropna(axis=1, how="all")
    feature_df = df.select_dtypes(include=[np.number]).copy()
    y = feature_df.pop("target")
    X_train, X_test, y_train, y_test = train_test_split(
        feature_df, y, test_size=cfg.test_size, random_state=cfg.random_state
    )

    clf = DecisionTreeClassifier(max_depth=cfg.max_depth, random_state=cfg.random_state)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    metrics = _summarize_training(cfg, matches, records, clf, X_train, X_test, y_train, y_test, acc)
    save_model(clf)
    save_metadata(metrics)
    return metrics
