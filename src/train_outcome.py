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
from sklearn.impute import SimpleImputer
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import Pipeline

try:
    from xgboost import XGBClassifier  # type: ignore
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "xgboost is required for train_outcome.py. Install it via `pip install xgboost`."
    ) from exc

LOGGER = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

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


@dataclass(slots=True)
class SplitConfig:
    train_end_year: int
    val_end_year: int
    auto_bounds: bool
    strategy: str  # "year" ou "row_split"
    row_split_year: int | None = None
    row_split_test_year: int | None = None


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train TieBreaker outcome model")
    parser.add_argument("--data", required=True, help="Chemin vers le dataset Parquet A/B")
    parser.add_argument("--train-end-year", type=int, default=None, help="Dernière année incluse dans le train (auto si absent)")
    parser.add_argument("--val-end-year", type=int, default=None, help="Dernière année incluse dans la validation (auto si absent)")
    parser.add_argument("--model-out", default="models/outcome_model.pkl", help="Chemin de sortie du modèle (pkl)")
    parser.add_argument("--report-out", default="reports/train_metrics.json", help="Chemin du rapport JSON")
    parser.add_argument("--seed", type=int, default=42, help="Seed aléatoire")
    return parser.parse_args(argv)


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


def get_available_years(df: pd.DataFrame) -> list[int]:
    years = sorted(df["year"].dropna().unique().tolist())
    if not years:
        raise ValueError("Aucune année valide détectée dans tourney_date")
    LOGGER.info(
        "Années disponibles (%d) — de %s à %s",
        len(years),
        years[0],
        years[-1],
    )
    return years


def resolve_split_config(years: list[int], train_arg: int | None, val_arg: int | None) -> SplitConfig:
    auto_bounds = train_arg is None and val_arg is None
    if auto_bounds:
        if len(years) >= 3:
            cfg = SplitConfig(
                train_end_year=years[-3],
                val_end_year=years[-2],
                auto_bounds=True,
                strategy="year",
            )
            LOGGER.info(
                "Split auto: train <= %s, val = %s, test >= %s",
                cfg.train_end_year,
                cfg.val_end_year,
                years[-1],
            )
            return cfg
        if len(years) == 2:
            cfg = SplitConfig(
                train_end_year=years[0],
                val_end_year=years[0],
                auto_bounds=True,
                strategy="row_split",
                row_split_year=years[0],
                row_split_test_year=years[1],
            )
            LOGGER.info(
                "Split auto (2 années): train/val sur %s (split lignes), test sur %s",
                years[0],
                years[1],
            )
            return cfg
        raise ValueError("Pas assez d'années pour créer une validation (il faut >= 2)")

    train_end = train_arg if train_arg is not None else years[0]
    val_end = val_arg if val_arg is not None else years[-1]
    if val_end < train_end:
        raise ValueError(
            f"val_end_year ({val_end}) doit être >= train_end_year ({train_end})."
        )
    LOGGER.info(
        "Split défini par l'utilisateur: train <= %s, val <= %s",
        train_end,
        val_end,
    )
    return SplitConfig(
        train_end_year=train_end,
        val_end_year=val_end,
        auto_bounds=False,
        strategy="year",
    )


def _build_row_split_masks(
    df: pd.DataFrame,
    first_year: int,
    test_year: int,
    seed: int,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    first_idx = df.index[df["year"] == first_year].to_numpy()
    if len(first_idx) < 2:
        raise ValueError(
            f"Impossible de créer une validation: l'année {first_year} ne contient pas assez de matches."
        )
    rng = np.random.default_rng(seed)
    perm = rng.permutation(first_idx)
    split_pt = max(1, int(round(len(perm) * 0.8)))
    if split_pt >= len(perm):
        split_pt = len(perm) - 1
    train_idx = perm[:split_pt]
    val_idx = perm[split_pt:]
    mask_train = pd.Series(False, index=df.index)
    mask_val = pd.Series(False, index=df.index)
    mask_train.loc[train_idx] = True
    mask_val.loc[val_idx] = True
    mask_test = df["year"] == test_year
    if mask_test.sum() == 0:
        raise ValueError(f"Aucune donnée pour l'année test {test_year}.")
    return mask_train, mask_val, mask_test


def build_splits(
    df: pd.DataFrame,
    feature_cols: list[str],
    split_cfg: SplitConfig,
    years: list[int],
    seed: int,
) -> DatasetSplits:
    if split_cfg.strategy == "row_split":
        assert split_cfg.row_split_year is not None and split_cfg.row_split_test_year is not None
        mask_train, mask_val, mask_test = _build_row_split_masks(
            df,
            split_cfg.row_split_year,
            split_cfg.row_split_test_year,
            seed,
        )
    else:
        mask_train = df["year"] <= split_cfg.train_end_year
        mask_val = (df["year"] > split_cfg.train_end_year) & (df["year"] <= split_cfg.val_end_year)
        mask_test = df["year"] > split_cfg.val_end_year

    n_train = int(mask_train.sum())
    n_val = int(mask_val.sum())
    n_test = int(mask_test.sum())
    LOGGER.info("Répartition des splits — train: %d, val: %d, test: %d", n_train, n_val, n_test)

    if n_train == 0 or n_val == 0 or n_test == 0:
        msg = (
            f"Invalid year split: n_train={n_train}, n_val={n_val}, n_test={n_test}. "
            f"Années dataset: {years[0]}-{years[-1]} (config train_end={split_cfg.train_end_year}, val_end={split_cfg.val_end_year})."
        )
        if split_cfg.auto_bounds:
            raise ValueError("Impossible de construire un split automatique valide. " + msg)
        raise ValueError(msg + " Vérifiez --train-end-year/--val-end-year.")

    X = df[feature_cols].replace({np.inf: np.nan, -np.inf: np.nan})
    y = df["y"]

    return DatasetSplits(
        X_train=X[mask_train],
        y_train=y[mask_train],
        X_val=X[mask_val],
        y_val=y[mask_val],
        X_test=X[mask_test],
        y_test=y[mask_test],
    )


def init_classifier(seed: int) -> Tuple[Any, str]:
    LOGGER.info("Utilisation du modèle XGBoost")
    model = XGBClassifier(
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
    imputer: SimpleImputer,
    splits: DatasetSplits,
) -> Tuple[Any, np.ndarray | None, np.ndarray | None]:
    X_train = imputer.fit_transform(splits.X_train)
    y_train = splits.y_train.to_numpy()
    X_val_imp = imputer.transform(splits.X_val) if len(splits.X_val) > 0 else None
    y_val_arr = splits.y_val.to_numpy() if len(splits.y_val) > 0 else None

    eval_set: list[tuple[np.ndarray, np.ndarray]] = [(X_train, y_train)]
    fit_kwargs: Dict[str, Any] = {"eval_set": eval_set, "verbose": False}
    if X_val_imp is not None and y_val_arr is not None and len(y_val_arr) > 0:
        eval_set.append((X_val_imp, y_val_arr))
        fit_kwargs["early_stopping_rounds"] = 50
    else:
        LOGGER.warning("Pas de validation pour l'early stopping — entraînement sans early stopping.")

    try:
        model.fit(X_train, y_train, **fit_kwargs)
    except TypeError as exc:
        if "early_stopping_rounds" in str(exc):
            LOGGER.warning("XGBoost ne supporte pas early_stopping_rounds — nouvel entraînement sans early stopping.")
            fit_kwargs.pop("early_stopping_rounds", None)
            model.fit(X_train, y_train, **fit_kwargs)
        else:
            raise
    return model, X_val_imp, y_val_arr


def calibrate_model(
    base_model: Any,
    X_cal: np.ndarray | None,
    y_cal: np.ndarray | None,
) -> Tuple[Any, str | None]:
    if X_cal is None or y_cal is None or len(y_cal) == 0:
        LOGGER.warning("Pas de set de validation: calibration ignorée.")
        return base_model, None
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
    years = get_available_years(df)
    split_cfg = resolve_split_config(years, args.train_end_year, args.val_end_year)
    LOGGER.info(
        "Bornes retenues — train_end_year=%s, val_end_year=%s (auto=%s, stratégie=%s)",
        split_cfg.train_end_year,
        split_cfg.val_end_year,
        split_cfg.auto_bounds,
        split_cfg.strategy,
    )
    splits = build_splits(df, feature_cols, split_cfg, years, args.seed)

    imputer = SimpleImputer(strategy="median")
    base_model, model_type = init_classifier(args.seed)
    trained_model, X_cal, y_cal = train_classifier(base_model, imputer, splits)
    calibrated_model, calibration_method = calibrate_model(trained_model, X_cal, y_cal)
    pipeline = build_final_pipeline(imputer, calibrated_model)

    metrics = {
        "train": evaluate_split(pipeline, splits.X_train, splits.y_train),
        "val": evaluate_split(pipeline, splits.X_val, splits.y_val),
        "test": evaluate_split(pipeline, splits.X_test, splits.y_test),
        "config": {
            "train_end_year": split_cfg.train_end_year,
            "val_end_year": split_cfg.val_end_year,
            "n_features": len(feature_cols),
            "model_type": model_type,
            "calibration_method": calibration_method,
            "split_strategy": split_cfg.strategy,
            "auto_bounds": split_cfg.auto_bounds,
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
        "train_end_year": artifacts.metrics["config"]["train_end_year"],
        "val_end_year": artifacts.metrics["config"]["val_end_year"],
        "model_type": artifacts.model_type,
        "calibration_method": artifacts.calibration_method,
        "split_strategy": artifacts.metrics["config"].get("split_strategy"),
        "auto_bounds": artifacts.metrics["config"].get("auto_bounds"),
        "created_at": datetime.now(UTC).isoformat(),
    }
    save_model(model_path, bundle)
    save_metrics(report_path, artifacts.metrics)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
