"""One-off outcome prediction utilities for TieBreaker."""
from __future__ import annotations

from dataclasses import dataclass
from difflib import get_close_matches
from pathlib import Path
from typing import Any, Dict, Optional

import joblib
import numpy as np
import pandas as pd

from .build_dataset import (
    PlayerLookup,
    add_one_hot_features,
    canonicalize_ab,
    normalize_name,
    prepare_players,
    prepare_rankings,
)
from .models import DataHub

try:  # optional recent form enrichments
    from .features_recent import add_recent_form_features
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    def add_recent_form_features(matches_df: pd.DataFrame, dataset: pd.DataFrame, **_: Any) -> pd.DataFrame:
        return dataset


@dataclass(slots=True)
class PredictRequest:
    p1_name: str
    p2_name: str
    date: Optional[pd.Timestamp] = None
    surface: str = "Hard"
    round: str = "R32"
    best_of: Optional[int] = None
    tourney_name: str = "Prediction"
    tourney_level: str = ""
    data_root: str = "data"
    model_path: str = "models/outcome_model_xgb.pkl"


@dataclass(slots=True)
class PredictResult:
    A_name: str
    B_name: str
    p_A_win: float
    p_B_win: float
    p_p1_win: float
    p_p2_win: float
    canonical_A_is_p1: bool
    meta: Dict[str, Any]


def predict_outcome(req: PredictRequest) -> PredictResult:
    bundle = joblib.load(req.model_path)
    model = bundle["model"]
    feature_cols: list[str] = bundle["features"]
    train_end_year = bundle.get("train_end_year")
    val_end_year = bundle.get("val_end_year")
    target_date = _resolve_target_date(req.date, val_end_year or train_end_year)

    hub = DataHub(Path(req.data_root))
    players_df_raw = hub.load_players()
    players_df, lookup = prepare_players(players_df_raw)
    rankings_df = prepare_rankings(hub.load_rankings())

    p1_id, p1_resolved = _resolve_player(players_df, req.p1_name)
    p2_id, p2_resolved = _resolve_player(players_df, req.p2_name)

    base_row = _build_base_feature_row(
        p1_id,
        p1_resolved,
        p2_id,
        p2_resolved,
        target_date,
        req,
        rankings_df,
        lookup,
    )
    dataset = pd.DataFrame([base_row])
    dataset = add_one_hot_features(dataset)

    matches_df = hub.load_matches().copy()
    matches_df["tourney_date"] = pd.to_datetime(matches_df["tourney_date"], errors="coerce")
    matches_df = matches_df[matches_df["tourney_date"] < target_date]
    dataset = add_recent_form_features(matches_df, dataset)

    dataset = dataset.fillna(np.nan)
    row_values = dataset.iloc[0]
    A_name = row_values["A_name"]
    B_name = row_values["B_name"]

    for col in feature_cols:
        if col not in dataset.columns:
            dataset[col] = 0.0

    X = dataset[feature_cols]
    probs = model.predict_proba(X)[0]
    p_A_win = float(probs[1])
    p_B_win = 1.0 - p_A_win

    canonical_A_is_p1 = normalize_name(A_name) == normalize_name(p1_resolved)
    if canonical_A_is_p1:
        p_p1_win = p_A_win
        p_p2_win = p_B_win
    else:
        p_p1_win = p_B_win
        p_p2_win = p_A_win

    meta = {
        "model_type": bundle.get("model_type"),
        "train_end_year": train_end_year,
        "val_end_year": val_end_year,
        "feature_count": len(feature_cols),
        "target_date": target_date.date(),
        "surface": row_values.get("surface"),
        "round": row_values.get("round"),
        "best_of": row_values.get("best_of"),
        "p1_resolved": p1_resolved,
        "p2_resolved": p2_resolved,
    }

    return PredictResult(
        A_name=A_name,
        B_name=B_name,
        p_A_win=p_A_win,
        p_B_win=p_B_win,
        p_p1_win=p_p1_win,
        p_p2_win=p_p2_win,
        canonical_A_is_p1=canonical_A_is_p1,
        meta=meta,
    )


def _resolve_target_date(date_value: Optional[pd.Timestamp], default_year: Optional[int]) -> pd.Timestamp:
    if date_value is not None:
        return pd.to_datetime(date_value).normalize()
    if default_year:
        return pd.Timestamp(year=default_year, month=12, day=31)
    return pd.Timestamp.utcnow().normalize()


def _resolve_player(players_df: pd.DataFrame, query: str) -> tuple[int, str]:
    if "full_name" not in players_df.columns:
        raise ValueError("Le fichier joueurs ne contient pas la colonne full_name.")
    candidates = players_df["full_name"].astype(str)
    normalized = candidates.map(_norm)
    target = _norm(query)
    match_df = players_df[normalized == target]
    if not match_df.empty:
        row = match_df.iloc[0]
    else:
        best = get_close_matches(query, candidates.tolist(), n=1, cutoff=0.7)
        if not best:
            raise ValueError(f"Joueur introuvable: {query}")
        row = players_df[candidates == best[0]].iloc[0]
    player_id = row.get("player_id")
    if pd.isna(player_id):
        raise ValueError(f"Identifiant joueur manquant pour {row['full_name']}")
    return int(player_id), str(row["full_name"])


def _build_base_feature_row(
    p1_id: int,
    p1_name: str,
    p2_id: int,
    p2_name: str,
    target_date: pd.Timestamp,
    req: PredictRequest,
    rankings_df: pd.DataFrame,
    lookup: PlayerLookup,
) -> Dict[str, Any]:
    match_row = {
        "winner_id": p1_id,
        "winner_name": p1_name,
        "loser_id": p2_id,
        "loser_name": p2_name,
        "tourney_date": target_date,
        "surface": _clean_surface(req.surface),
        "round": _clean_round(req.round),
        "best_of": req.best_of,
        "tourney_name": req.tourney_name,
        "tourney_level": req.tourney_level,
    }
    features = canonicalize_ab(match_row, rankings_df, lookup)
    return features


def _norm(value: str) -> str:
    return " ".join(str(value).strip().split()).casefold()


def _clean_surface(surface: Optional[str]) -> str:
    if not surface:
        return "Hard"
    surface_normalized = surface.strip().title()
    if surface_normalized not in {"Hard", "Clay", "Grass", "Carpet"}:
        return "Hard"
    return surface_normalized


def _clean_round(round_value: Optional[str]) -> str:
    if not round_value:
        return "R32"
    return round_value.strip().upper()
