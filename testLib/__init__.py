"""Convenience exports for the testLib ML toolkit."""
from .model import TrainConfig, train_model, predict_match
from .model.predictor import predict
from .features import feature_vector_from_matches
from .data import load_matches, latest_match_for_player

__all__ = [
    "TrainConfig",
    "train_model",
    "predict",
    "predict_match",
    "feature_vector_from_matches",
    "load_matches",
    "latest_match_for_player",
]
