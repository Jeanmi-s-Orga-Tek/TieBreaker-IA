"""One-off outcome prediction utilities for TieBreaker."""
from __future__ import annotations

from dataclasses import dataclass, replace
from difflib import get_close_matches
from pathlib import Path
from typing import Any, Dict, Optional

import joblib
import numpy as np
import pandas as pd

try:
    from .build_dataset import (
        PlayerLookup,
        add_derived_features,
        add_one_hot_features,
        canonicalize_ab,
        normalize_name,
        prepare_players,
        prepare_rankings,
    )
    from .models import DataHub
except ImportError:  # pragma: no cover - allow running via src on sys.path
    from build_dataset import (
        PlayerLookup,
        add_derived_features,
        add_one_hot_features,
        canonicalize_ab,
        normalize_name,
        prepare_players,
        prepare_rankings,
    )
    from models import DataHub

try:
    from .features_recent import add_recent_form_features, RECENT_FEATURE_MAP
except ModuleNotFoundError:
    try:
        from features_recent import add_recent_form_features, RECENT_FEATURE_MAP
    except ModuleNotFoundError:  # pragma: no cover - optional dependency
        RECENT_FEATURE_MAP = {}

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

    recent_feature_cols = {"recent_form_missing_A", "recent_form_missing_B", *RECENT_FEATURE_MAP.values()}
    needs_recent_form = any(col in feature_cols for col in recent_feature_cols)

    matches_df = None
    if needs_recent_form or req.best_of is None:
        matches_df = _load_matches_for_prediction(hub, target_date, {p1_id, p2_id})

    best_of_value, best_of_source = _resolve_best_of(
        req,
        matches_df if matches_df is not None else pd.DataFrame(),
        p1_resolved,
        p2_resolved,
        target_date,
    )
    req_features = replace(req, best_of=best_of_value)

    base_row = _build_base_feature_row(
        p1_id,
        p1_resolved,
        p2_id,
        p2_resolved,
        target_date,
        req_features,
        rankings_df,
        lookup,
    )
    dataset = pd.DataFrame([base_row])
    dataset = add_one_hot_features(dataset)
    dataset = add_derived_features(dataset)

    if needs_recent_form and matches_df is not None:
        matches_history = matches_df[matches_df["tourney_date"] < target_date]
        dataset = add_recent_form_features(matches_history, dataset)

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
        "best_of_source": best_of_source,
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


def _filter_matches_for_players(matches_df: pd.DataFrame, player_ids: set[int]) -> pd.DataFrame:
    if matches_df.empty or not player_ids:
        return matches_df
    if "winner_id" not in matches_df.columns or "loser_id" not in matches_df.columns:
        return matches_df
    filtered = matches_df.copy()
    filtered["winner_id"] = pd.to_numeric(filtered["winner_id"], errors="coerce")
    filtered["loser_id"] = pd.to_numeric(filtered["loser_id"], errors="coerce")
    mask = filtered["winner_id"].isin(player_ids) | filtered["loser_id"].isin(player_ids)
    return filtered[mask].reset_index(drop=True)


def _load_matches_for_prediction(
    hub: DataHub,
    target_date: pd.Timestamp,
    player_ids: set[int],
    window_years: int = 5,
) -> pd.DataFrame:
    if isinstance(target_date, pd.Timestamp) and pd.notna(target_date):
        end_year = int(target_date.year)
        start_year = end_year - max(window_years - 1, 0)
        years = list(range(start_year, end_year + 1))
        matches_df = hub.load_matches(years=years).copy()
    else:
        matches_df = hub.load_matches().copy()
    matches_df["tourney_date"] = pd.to_datetime(matches_df["tourney_date"], errors="coerce")
    return _filter_matches_for_players(matches_df, player_ids)


def _safe_int(value: Any) -> Optional[int]:
    try:
        if pd.isna(value):
            return None
    except TypeError:
        pass
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _default_best_of(req: PredictRequest) -> int:
    level = str(req.tourney_level or "").strip().upper()
    if level == "G":
        return 5
    name = str(req.tourney_name or "").strip().casefold()
    for slam in ("wimbledon", "roland garros", "us open", "australian open"):
        if slam in name:
            return 5
    return 3


def _infer_best_of_from_matches(
    matches_df: pd.DataFrame,
    p1_name: str,
    p2_name: str,
    target_date: pd.Timestamp,
    surface: Optional[str],
    round_value: Optional[str],
) -> Optional[int]:
    if matches_df.empty or pd.isna(target_date):
        return None
    if "tourney_date" not in matches_df.columns:
        return None
    if "winner_name" not in matches_df.columns or "loser_name" not in matches_df.columns:
        return None

    target_date = pd.to_datetime(target_date).normalize()
    start = target_date - pd.Timedelta(days=21)
    end = target_date + pd.Timedelta(days=1)
    candidates = matches_df[matches_df["tourney_date"].between(start, end)]
    if candidates.empty:
        return None

    p1_key = normalize_name(p1_name)
    p2_key = normalize_name(p2_name)
    winner_key = candidates["winner_name"].astype(str).map(normalize_name)
    loser_key = candidates["loser_name"].astype(str).map(normalize_name)
    player_mask = ((winner_key == p1_key) & (loser_key == p2_key)) | ((winner_key == p2_key) & (loser_key == p1_key))
    candidates = candidates[player_mask]
    if candidates.empty:
        return None

    if round_value and "round" in candidates.columns:
        round_norm = str(round_value).strip().upper()
        round_filtered = candidates[candidates["round"].astype(str).str.strip().str.upper() == round_norm]
        if not round_filtered.empty:
            candidates = round_filtered

    if surface and "surface" in candidates.columns:
        surface_norm = str(surface).strip().title()
        surface_filtered = candidates[candidates["surface"].astype(str).str.strip().str.title() == surface_norm]
        if not surface_filtered.empty:
            candidates = surface_filtered

    candidates = candidates.copy()
    candidates["__date_delta"] = (candidates["tourney_date"] - target_date).abs()
    candidates = candidates.sort_values(["__date_delta"])
    return _safe_int(candidates.iloc[0].get("best_of"))


def _resolve_best_of(
    req: PredictRequest,
    matches_df: pd.DataFrame,
    p1_name: str,
    p2_name: str,
    target_date: pd.Timestamp,
) -> tuple[int, str]:
    direct = _safe_int(req.best_of)
    if direct is not None:
        return direct, "provided"
    inferred = _infer_best_of_from_matches(matches_df, p1_name, p2_name, target_date, req.surface, req.round)
    if inferred is not None:
        return inferred, "match_lookup"
    return _default_best_of(req), "default"


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
