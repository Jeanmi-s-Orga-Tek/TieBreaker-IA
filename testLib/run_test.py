"""Minimal demo for training and inference."""
from __future__ import annotations

from .model import train_model, TrainConfig
from .model.predictor import predict
from .data import dataset_exists


if __name__ == "__main__":
    if not dataset_exists():
        raise FileNotFoundError(
            "No dataset found in testLib/data/data.csv. Populate it and rerun."
        )

    metrics = train_model(TrainConfig(max_depth=3))
    print(f"Training metrics: {metrics}")

    sample_input = {"feature_1": 1.0, "feature_2": 2.0}
    prediction = predict(sample_input)
    print(f"Prediction for {sample_input}: {prediction}")
