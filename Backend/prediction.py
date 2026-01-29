"""
Prediction API endpoints for TieBreaker
"""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from pathlib import Path
from typing import Optional
import sys
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

src_path = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(src_path))

from Backend.model_loader import load_model_cached, is_model_loaded

try:
    from predict_outcome import PredictRequest, predict_outcome as _predict_outcome
    PREDICTION_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Could not import predict_outcome: {e}")
    PREDICTION_AVAILABLE = False
    _predict_outcome = None

router = APIRouter()


class PredictionInput(BaseModel):
    player1_name: str
    player2_name: str
    surface: str = "Hard"
    round: str = "R32"
    tourney_level: str = ""
    tournament: Optional[str] = None
    date: Optional[str] = None
    year: Optional[int] = None
    all_years: bool = False


class PredictionOutput(BaseModel):
    player1: str
    player2: str
    winner_prediction: str
    probability: float
    confidence: str
    details: dict


def predict_outcome_cached(req):
    """
    Wrapper around predict_outcome that uses cached model
    """
    import joblib
    import numpy as np
    import pandas as pd
    from build_dataset import (
        PlayerLookup,
        add_one_hot_features,
        canonicalize_ab,
        normalize_name,
        prepare_players,
        prepare_rankings,
    )
    from models import DataHub

    try:
        from features_recent import add_recent_form_features
    except ModuleNotFoundError:
        def add_recent_form_features(matches_df, dataset, **_):
            return dataset

    bundle = load_model_cached(req.model_path)
    model = bundle["model"]
    feature_cols = bundle["features"]
    train_end_year = bundle.get("train_end_year")
    val_end_year = bundle.get("val_end_year")

    target_date = req.date
    if target_date is None:
        if val_end_year:
            target_date = pd.Timestamp(year=val_end_year, month=12, day=31)
        elif train_end_year:
            target_date = pd.Timestamp(year=train_end_year, month=12, day=31)
        else:
            target_date = pd.Timestamp.utcnow().normalize()
    else:
        target_date = pd.to_datetime(target_date).normalize()

    hub = DataHub(Path(req.data_root))
    players_df_raw = hub.load_players()
    players_df, lookup = prepare_players(players_df_raw)
    rankings_df = prepare_rankings(hub.load_rankings())

    from difflib import get_close_matches

    def _resolve_player(players_df, query):
        if "full_name" not in players_df.columns:
            raise ValueError("Le fichier joueurs ne contient pas la colonne full_name.")
        candidates = players_df["full_name"].astype(str)
        normalized = candidates.map(lambda x: " ".join(str(x).strip().split()).casefold())
        target = " ".join(str(query).strip().split()).casefold()
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

    p1_id, p1_resolved = _resolve_player(players_df, req.p1_name)
    p2_id, p2_resolved = _resolve_player(players_df, req.p2_name)

    def _clean_surface(surface):
        if not surface:
            return "Hard"
        surface_normalized = surface.strip().title()
        if surface_normalized not in {"Hard", "Clay", "Grass", "Carpet"}:
            return "Hard"
        return surface_normalized

    def _clean_round(round_value):
        if not round_value:
            return "R32"
        return round_value.strip().upper()

    match_row = {
        "winner_id": p1_id,
        "winner_name": p1_resolved,
        "loser_id": p2_id,
        "loser_name": p2_resolved,
        "tourney_date": target_date,
        "surface": _clean_surface(req.surface),
        "round": _clean_round(req.round),
        "best_of": req.best_of,
        "tourney_name": req.tourney_name,
        "tourney_level": req.tourney_level,
    }
    features = canonicalize_ab(match_row, rankings_df, lookup)
    dataset = pd.DataFrame([features])
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
        "train_end_year": int(train_end_year) if train_end_year else None,
        "val_end_year": int(val_end_year) if val_end_year else None,
        "feature_count": len(feature_cols),
        "target_date": str(target_date.date()),
        "surface": str(row_values.get("surface")),
        "round": str(row_values.get("round")),
        "best_of": int(row_values.get("best_of")) if pd.notna(row_values.get("best_of")) else None,
        "p1_resolved": str(p1_resolved),
        "p2_resolved": str(p2_resolved),
    }

    class PredictResult:
        pass

    result = PredictResult()
    result.A_name = str(A_name)
    result.B_name = str(B_name)
    result.p_A_win = float(p_A_win)
    result.p_B_win = float(p_B_win)
    result.p_p1_win = float(p_p1_win)
    result.p_p2_win = float(p_p2_win)
    result.canonical_A_is_p1 = bool(canonical_A_is_p1)
    result.meta = meta

    return result


@router.post("/api/predict")
async def predict_match(input_data: PredictionInput) -> PredictionOutput:
    """
    Predict the outcome of a tennis match between two players
    """
    if not PREDICTION_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail="Prediction service not available. Required modules may not be loaded."
        )

    try:
        from predict_outcome import PredictRequest

        target_date = None
        if input_data.date:
            target_date = input_data.date
        elif input_data.year:
            target_date = f"{input_data.year}-12-31"

        req = PredictRequest(
            p1_name=input_data.player1_name,
            p2_name=input_data.player2_name,
            surface=input_data.surface,
            round=input_data.round,
            tourney_level=input_data.tourney_level,
            tourney_name=input_data.tournament or "Prediction",
            date=target_date,
            data_root="data",
            model_path="models/outcome_model_xgb.pkl"
        )

        result = predict_outcome_cached(req)

        if result.p_p1_win >= result.p_p2_win:
            winner = str(result.A_name if result.canonical_A_is_p1 else result.B_name)
            probability = float(result.p_p1_win)
        else:
            winner = str(result.B_name if result.canonical_A_is_p1 else result.A_name)
            probability = float(result.p_p2_win)

        if probability >= 0.7:
            confidence = "Très élevée"
        elif probability >= 0.6:
            confidence = "Élevée"
        elif probability >= 0.55:
            confidence = "Moyenne"
        else:
            confidence = "Faible"

        return PredictionOutput(
            player1=str(result.A_name if result.canonical_A_is_p1 else result.B_name),
            player2=str(result.B_name if result.canonical_A_is_p1 else result.A_name),
            winner_prediction=winner,
            probability=probability,
            confidence=confidence,
            details={
                "p1_win_probability": float(result.p_p1_win),
                "p2_win_probability": float(result.p_p2_win),
                "model_info": result.meta
            }
        )

    except ValueError as e:
        logger.error(f"ValueError in prediction: {e}")
        raise HTTPException(status_code=404, detail=f"Erreur: {str(e)}")
    except FileNotFoundError as e:
        logger.error(f"FileNotFoundError: {e}")
        raise HTTPException(status_code=503, detail=f"Fichier manquant: {str(e)}")
    except Exception as e:
        logger.error(f"Unexpected error in prediction: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Erreur interne: {str(e)}")


@router.get("/api/model-status")
async def model_status():
    """
    Check if the model is loaded and ready
    """
    return {
        "model_loaded": is_model_loaded(),
        "prediction_available": PREDICTION_AVAILABLE,
    }

