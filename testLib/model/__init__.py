"""Model namespace exports."""
from .trainer import TrainConfig, train_model
from .predictor import predict, predict_match

__all__ = ["TrainConfig", "train_model", "predict", "predict_match"]

