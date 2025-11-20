"""Model serialization helpers."""
from __future__ import annotations

from pathlib import Path
from typing import Any
import json
from datetime import datetime

import joblib

MODEL_PATH = Path(__file__).resolve().parent / "model.pkl"
METADATA_PATH = Path(__file__).resolve().parent / "model_meta.json"


def save_model(model: Any, path: Path | None = None) -> Path:
    """Serialize the estimator to disk."""
    model_path = Path(path) if path else MODEL_PATH
    joblib.dump(model, model_path)
    return model_path


def load_model(path: Path | None = None) -> Any:
    """Load a previously persisted estimator."""
    model_path = Path(path) if path else MODEL_PATH
    if not model_path.is_file():
        raise FileNotFoundError(f"Model missing: {model_path}")
    return joblib.load(model_path)


def save_metadata(payload: dict, path: Path | None = None) -> Path:
    """Persist training metadata (metrics, timestamps, etc.)."""
    metadata_path = Path(path) if path else METADATA_PATH
    payload = {**payload, "saved_at": datetime.utcnow().isoformat()}
    metadata_path.write_text(json.dumps(payload, indent=2))
    return metadata_path


def load_metadata(path: Path | None = None) -> dict | None:
    """Retrieve stored metadata if present."""
    metadata_path = Path(path) if path else METADATA_PATH
    if not metadata_path.is_file():
        return None
    return json.loads(metadata_path.read_text())
