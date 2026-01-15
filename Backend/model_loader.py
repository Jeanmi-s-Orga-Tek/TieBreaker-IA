"""
Model loader with caching and error handling
"""
import joblib
import logging
from pathlib import Path
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)

# Global cache for the loaded model
_MODEL_CACHE: Optional[Dict[str, Any]] = None
_MODEL_PATH: Optional[Path] = None


def load_model_cached(model_path: str = "models/outcome_model_xgb.pkl") -> Dict[str, Any]:
    """
    Load the ML model with caching to avoid reloading on every request.

    Args:
        model_path: Path to the model file

    Returns:
        Dictionary containing the model and metadata

    Raises:
        FileNotFoundError: If model file doesn't exist
        Exception: If model loading fails
    """
    global _MODEL_CACHE, _MODEL_PATH

    model_file = Path(model_path)

    # Return cached model if available and path hasn't changed
    if _MODEL_CACHE is not None and _MODEL_PATH == model_file:
        logger.info(f"Using cached model from {model_path}")
        return _MODEL_CACHE

    # Check if model file exists
    if not model_file.exists():
        raise FileNotFoundError(
            f"Model file not found: {model_path}. "
            f"Please ensure the model is trained and saved at this location."
        )

    logger.info(f"Loading model from {model_path}...")

    try:
        # Load the model
        bundle = joblib.load(model_path)

        # Validate model structure
        if not isinstance(bundle, dict):
            raise ValueError("Model file does not contain a valid dictionary")

        if "model" not in bundle:
            raise ValueError("Model file does not contain 'model' key")

        if "features" not in bundle:
            raise ValueError("Model file does not contain 'features' key")

        # Cache the model
        _MODEL_CACHE = bundle
        _MODEL_PATH = model_file

        logger.info(
            f"Model loaded successfully. "
            f"Type: {bundle.get('model_type', 'unknown')}, "
            f"Features: {len(bundle.get('features', []))}"
        )

        return bundle

    except Exception as e:
        logger.error(f"Failed to load model from {model_path}: {e}")
        raise Exception(f"Failed to load model: {str(e)}")


def clear_model_cache():
    """Clear the cached model (useful for testing or reloading)"""
    global _MODEL_CACHE, _MODEL_PATH
    _MODEL_CACHE = None
    _MODEL_PATH = None
    logger.info("Model cache cleared")


def is_model_loaded() -> bool:
    """Check if a model is currently loaded in cache"""
    return _MODEL_CACHE is not None

