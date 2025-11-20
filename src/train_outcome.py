"""Model training entrypoint for TieBreaker outcome prediction."""
from __future__ import annotations

import argparse
import json
import logging
import random
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Dict, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import Pipeline

LOGGER = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

try:  # Optional dependency
    from xgboost import XGBClassifier  # type: ignore

    HAS_XGB = True
except ImportError:  # pragma: no cover - only triggered when xgboost missing
    XGBClassifier = None  # type: ignore
    HAS_XGB = False

EXCLUDE_COLUMNS = {
    "y",
    "A_name",
    "B_name",
    "A_player_id",
    "B_player_id",
    "winner_name_raw",
    "loser_name_raw",
    "tourney_name",
    "tourney_date",
    "year",
}


@dataclass(slots=True)
class DatasetSplits:
    X_train: pd.DataFrame
    y_train: pd.Series
    X_val: pd.DataFrame
    y_val: pd.Series
    X_test: pd.DataFrame
    y_test: pd.Series


@dataclass(slots=True)
class TrainingArtifacts:
    model: Pipeline
    metrics: Dict[str, Any]
    feature_columns: list[str]
    model_type: str
    calibration_method: str | None


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train TieBreaker outcome model")
    parser.add_argument("--data", required=True, help="Chemin vers le dataset Parquet A/B")
    parser.add_argument("--train-end-year", type=int, default=2021, help="Dernière année incluse dans le train (défaut: 2021)")
    parser.add_argument("--val-end-year", type=int, default=2023, help="Dernière année incluse dans la validation (défaut: 2023)")
    parser.add_argument("--model-out", default="models/outcome_model.pkl", help="Chemin de sortie du modèle (pkl)")
    parser.add_argument("--report-out", default="reports/train_metrics.json", help="Chemin du rapport JSON")
    parser.add_argument("--seed", type=int, default=42, help="Seed aléatoire")
    parser.add_argument("--force-xgb", action="store_true", help="Forcer l'utilisation de XGBoost (erreur si indisponible)")
    parser.add_argument("--force-gbdt", action="store_true", help="Forcer GradientBoostingClassifier même si XGBoost est installé")
    args = parser.parse_args(argv)
    if args.force_xgb and args.force_gbdt:
        parser.error("Choisir --force-xgb ou --force-gbdt, pas les deux.")
    if args.force_xgb and not HAS_XGB:
        parser.error("xgboost n'est pas installé, impossible d'utiliser --force-xgb.")
    if args.val_end_year < args.train_end_year:
        parser.error("val_end_year doit être >= train_end_year")
    return args


def load_dataset(data_path: Path) -> pd.DataFrame:
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset introuvable: {data_path}")
    LOGGER.info("Chargement du dataset %s", data_path)
    df = pd.read_parquet(data_path)
    if "y" not in df.columns:
        raise ValueError("La colonne 'y' est absente du dataset")
    df["tourney_date"] = pd.to_datetime(df["tourney_date"], errors="coerce")
    df = df.dropna(subset=["tourney_date", "y"]).reset_index(drop=True)
    df["y"] = pd.to_numeric(df["y"], errors="coerce")
    df = df.dropna(subset=["y"])
    df["y"] = df["y"].astype(int)
    df["year"] = df["tourney_date"].dt.year
    return df


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    numeric_cols = df.select_dtypes(include=["number", "bool"]).columns
    features = [col for col in numeric_cols if col not in EXCLUDE_COLUMNS]
    if not features:
        raise ValueError("Aucune feature numérique disponible après exclusion")
    LOGGER.info("Features sélectionnées (%d): %s%s", len(features), features[:10], "..." if len(features) > 10 else "")
    return features


def build_splits(df: pd.DataFrame, train_end_year: int, val_end_year: int, feature_cols: list[str]) -> DatasetSplits:
    mask_train = df["year"] <= train_end_year
    mask_val = (df["year"] > train_end_year) & (df["year"] <= val_end_year)
    mask_test = df["year"] > val_end_year

    if mask_train.sum() == 0:
        fallback_year = int(df["year"].min())
        LOGGER.warning(
            "Aucune donnée dans le split train (<= %s). Fallback sur l'année minimale disponible %s.",
            train_end_year,
            fallback_year,
        )
        mask_train = df["year"] == fallback_year
        mask_val &= ~mask_train
        mask_test &= ~mask_train

    X = df[feature_cols].replace({np.inf: np.nan, -np.inf: np.nan})
    y = df["y"]

    splits = DatasetSplits(
        X_train=X[mask_train],
        y_train=y[mask_train],
        X_val=X[mask_val],
        y_val=y[mask_val],
        X_test=X[mask_test],
        y_test=y[mask_test],
    )

    LOGGER.info(
        "Répartition des splits — train: %d, val: %d, test: %d",
        len(splits.X_train),
        len(splits.X_val),
        len(splits.X_test),
    )
    if len(splits.X_val) == 0:
        LOGGER.warning("Split validation vide — réutilisation du train pour l'early stopping / calibration.")
    if len(splits.X_test) == 0:
        LOGGER.warning("Split test vide — les métriques test seront absentes.")
    return splits


def init_classifier(force_gbdt: bool, force_xgb: bool, seed: int) -> Tuple[Any, str]:
    if force_gbdt or (not HAS_XGB and not force_xgb):
        LOGGER.info("Utilisation du modèle GradientBoostingClassifier")
        model = GradientBoostingClassifier(n_estimators=400, learning_rate=0.05, max_depth=3, random_state=seed)
        return model, "gradient_boosting"
    LOGGER.info("Utilisation du modèle XGBoost")
    model = XGBClassifier(  # type: ignore[call-arg]
        objective="binary:logistic",
        eval_metric="logloss",
        n_estimators=1000,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=seed,
        n_jobs=-1,
    )
    return model, "xgboost"


def train_classifier(
    model: Any,
    model_type: str,
    imputer: SimpleImputer,
    splits: DatasetSplits,
) -> Tuple[Any, np.ndarray, np.ndarray]:
    X_train = imputer.fit_transform(splits.X_train)
    y_train = splits.y_train.to_numpy()

    X_cal = splits.X_val if len(splits.X_val) > 0 else splits.X_train
    y_cal = splits.y_val if len(splits.y_val) > 0 else splits.y_train
    X_cal_imp = imputer.transform(X_cal)

    if model_type == "xgboost":
        fit_kwargs: Dict[str, Any] = {"verbose": False}
        if len(X_cal_imp) > 0:
            fit_kwargs["eval_set"] = [(X_cal_imp, y_cal.to_numpy())]
            fit_kwargs["early_stopping_rounds"] = 50
        else:
            LOGGER.warning("Pas de données pour l'early stopping XGBoost.")
        model.fit(X_train, y_train, **fit_kwargs)
    else:
        model.fit(X_train, y_train)

    return model, X_cal_imp, y_cal.to_numpy()


def calibrate_model(
    base_model: Any,
    X_cal: np.ndarray,
    y_cal: np.ndarray,
) -> Tuple[Any, str | None]:
    if len(np.unique(y_cal)) < 2:
        LOGGER.warning("Impossible de calibrer (une seule classe). Modèle non calibré.")
        return base_model, None
    method = "isotonic" if len(y_cal) >= 1000 else "sigmoid"
    LOGGER.info("Calibration des probabilités (%s)", method)
    calibrator_kwargs: Dict[str, Any] = {"method": method, "cv": "prefit"}
    try:
        calibrator = CalibratedClassifierCV(estimator=base_model, **calibrator_kwargs)
    except TypeError:  # scikit-learn < 1.4 uses base_estimator
        calibrator = CalibratedClassifierCV(base_estimator=base_model, **calibrator_kwargs)  # type: ignore[arg-type]
    calibrator.fit(X_cal, y_cal)
    return calibrator, method


def evaluate_split(model: Pipeline, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
    if len(X) == 0:
        return {"auc": None, "logloss": None, "brier": None, "n": 0}
    probs = model.predict_proba(X)[:, 1]
    auc = roc_auc_score(y, probs) if len(np.unique(y)) > 1 else None
    logloss = log_loss(y, probs, labels=[0, 1])
    brier = brier_score_loss(y, probs)
    return {
        "auc": float(auc) if auc is not None else None,
        "logloss": float(logloss),
        "brier": float(brier),
        "n": int(len(y)),
    }


def build_final_pipeline(imputer: SimpleImputer, model: Any) -> Pipeline:
    return Pipeline([
        ("imputer", imputer),
        ("model", model),
    ])


def save_metrics(report_path: Path, metrics: Dict[str, Any]) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2)
    LOGGER.info("Rapport sauvegardé dans %s", report_path)


def save_model(model_path: Path, bundle: Dict[str, Any]) -> None:
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, model_path)
    LOGGER.info("Modèle sauvegardé dans %s", model_path)


def train_pipeline(args: argparse.Namespace) -> TrainingArtifacts:
    random.seed(args.seed)
    np.random.seed(args.seed)

    data_path = Path(args.data)
    df = load_dataset(data_path)
    feature_cols = get_feature_columns(df)
    splits = build_splits(df, args.train_end_year, args.val_end_year, feature_cols)

    imputer = SimpleImputer(strategy="median")
    base_model, model_type = init_classifier(args.force_gbdt, args.force_xgb, args.seed)
    trained_model, X_cal, y_cal = train_classifier(base_model, model_type, imputer, splits)
    calibrated_model, calibration_method = calibrate_model(trained_model, X_cal, y_cal)
    pipeline = build_final_pipeline(imputer, calibrated_model)

    metrics = {
        "train": evaluate_split(pipeline, splits.X_train, splits.y_train),
        "val": evaluate_split(pipeline, splits.X_val, splits.y_val),
        "test": evaluate_split(pipeline, splits.X_test, splits.y_test),
        "config": {
            "train_end_year": args.train_end_year,
            "val_end_year": args.val_end_year,
            "n_features": len(feature_cols),
            "model_type": model_type,
            "calibration_method": calibration_method,
        },
    }

    LOGGER.info("Métriques — train AUC: %s, val AUC: %s, test AUC: %s",
                metrics["train"]["auc"], metrics["val"]["auc"], metrics["test"]["auc"])

    return TrainingArtifacts(
        model=pipeline,
        metrics=metrics,
        feature_columns=feature_cols,
        model_type=model_type,
        calibration_method=calibration_method,
    )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    artifacts = train_pipeline(args)

    model_path = Path(args.model_out)
    report_path = Path(args.report_out)

    bundle = {
        "model": artifacts.model,
        "features": artifacts.feature_columns,
        "train_end_year": args.train_end_year,
        "val_end_year": args.val_end_year,
        "model_type": artifacts.model_type,
        "calibration_method": artifacts.calibration_method,
        "created_at": datetime.now(UTC).isoformat(),
    }
    save_model(model_path, bundle)
    save_metrics(report_path, artifacts.metrics)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
