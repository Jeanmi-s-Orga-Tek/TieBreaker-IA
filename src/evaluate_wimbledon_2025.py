"""Evaluate the XGBoost outcome model on Wimbledon 2025 matches."""
from __future__ import annotations

import argparse
import re
import sys
from datetime import datetime, UTC
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

try:
    from .build_dataset import (
        add_derived_features,
        add_one_hot_features,
        canonicalize_ab,
        prepare_players,
        prepare_rankings,
    )
    from .features_recent import add_recent_form_features
    from .models import DataHub
except ImportError:  # pragma: no cover - allow running via src on sys.path
    from build_dataset import (
        add_derived_features,
        add_one_hot_features,
        canonicalize_ab,
        prepare_players,
        prepare_rankings,
    )
    from features_recent import add_recent_form_features
    from models import DataHub


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate outcome_model_xgb_v4 on Wimbledon 2025 matches."
    )
    parser.add_argument(
        "--model-path",
        default="models/outcome_model_xgb_v4.pkl",
        help="Chemin du modele pickle/joblib.",
    )
    parser.add_argument(
        "--data-root",
        default="data",
        help="Racine des donnees ATP (players, rankings, matches historiques).",
    )
    parser.add_argument(
        "--matches",
        type=Path,
        default=None,
        help="CSV/Parquet contenant les matches Wimbledon 2025 (si absent: data/atp_matches/atp_matches_2025.csv).",
    )
    parser.add_argument("--tourney-name", default="Wimbledon", help="Nom du tournoi a filtrer.")
    parser.add_argument("--year", type=int, default=2025, help="Annee du tournoi.")
    parser.add_argument(
        "--tourney-level",
        default="G",
        help="Filtre sur tourney_level (ex: G). Laisser vide pour desactiver.",
    )
    parser.add_argument(
        "--history-years",
        type=int,
        default=None,
        help="Limiter l'historique aux N dernieres annees pour les features recentes.",
    )
    parser.add_argument(
        "--report-out",
        type=Path,
        default=Path("reports/eval_wimbledon_2025.json"),
        help="Sortie JSON des metriques.",
    )
    parser.add_argument(
        "--predictions-out",
        type=Path,
        default=None,
        help="CSV de sorties par match (probabilites + verdicts).",
    )
    return parser.parse_args(argv)


def _normalize_matches_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    cols = {c.lower(): c for c in df.columns}

    tdate = cols.get("tourney_date") or "tourney_date"
    if tdate in df.columns:
        def parse_date(v: Any):
            try:
                s = str(int(v))
                return datetime.strptime(s, "%Y%m%d").date()
            except Exception:
                try:
                    return pd.to_datetime(v, errors="coerce").date()
                except Exception:
                    return pd.NaT
        df["tourney_date"] = df[tdate].apply(parse_date)

    for key in [
        "winner_name",
        "loser_name",
        "tourney_name",
        "round",
        "score",
        "surface",
        "best_of",
        "winner_id",
        "loser_id",
        "tourney_level",
        "tourney_id",
        "match_num",
    ]:
        if key not in df.columns and key in cols:
            df = df.rename(columns={cols[key]: key})

    for key in ["winner_name", "loser_name", "tourney_name", "round", "score", "surface"]:
        if key in df.columns:
            df[key] = df[key].astype(str)
    return df


def _load_matches_file(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Fichier matches introuvable: {path}")
    if path.suffix.lower() in {".parquet", ".pq"}:
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path, low_memory=False)
    return _normalize_matches_df(df)


def _available_match_years(data_root: Path) -> list[int]:
    matches_dir = data_root / "atp_matches"
    years = []
    for path in matches_dir.glob("atp_matches_*.csv"):
        m = re.search(r"(\d{4})", path.name)
        if m:
            years.append(int(m.group(1)))
    return sorted(set(years))


def _load_history_matches(data_root: Path, max_year: int | None, history_years: int | None) -> pd.DataFrame:
    hub = DataHub(data_root)
    years = _available_match_years(data_root)
    if max_year is not None:
        years = [y for y in years if y <= max_year]
    if history_years is not None and years:
        years = years[-history_years:]
    matches = hub.load_matches(years=years) if years else hub.load_matches()
    return _normalize_matches_df(matches)


def _merge_history(history: pd.DataFrame, extra: pd.DataFrame) -> pd.DataFrame:
    if history is None or history.empty:
        return extra
    if extra is None or extra.empty:
        return history
    combined = pd.concat([history, extra], ignore_index=True)
    dedup_cols = [
        c for c in [
            "tourney_id",
            "match_num",
            "tourney_date",
            "winner_id",
            "loser_id",
            "winner_name",
            "loser_name",
        ] if c in combined.columns
    ]
    if dedup_cols:
        combined = combined.drop_duplicates(subset=dedup_cols)
    return combined.reset_index(drop=True)


def _extract_year(series: pd.Series) -> pd.Series:
    year = pd.to_datetime(series, errors="coerce").dt.year
    return year


def _filter_tournament(
    matches: pd.DataFrame,
    tourney_name: str,
    year: int,
    tourney_level: str | None,
) -> pd.DataFrame:
    if matches.empty:
        return matches
    if "tourney_name" in matches.columns:
        name_mask = matches["tourney_name"].astype(str).str.casefold().str.contains(tourney_name.casefold())
    else:
        name_mask = pd.Series(True, index=matches.index)

    if "tourney_date" in matches.columns:
        year_values = _extract_year(matches["tourney_date"])
    else:
        year_values = pd.Series([np.nan] * len(matches))
    if year_values.isna().all() and "tourney_id" in matches.columns:
        fallback = matches["tourney_id"].astype(str).str.slice(0, 4)
        year_values = pd.to_numeric(fallback, errors="coerce")
    year_mask = year_values == year if year_values.notna().any() else pd.Series(True, index=matches.index)
    mask = name_mask & year_mask
    if tourney_level:
        level_mask = matches.get("tourney_level")
        if level_mask is not None:
            mask &= level_mask.astype(str).str.upper().eq(tourney_level.upper())
    return matches[mask].reset_index(drop=True)


def _build_eval_dataset(
    matches: pd.DataFrame,
    history_matches: pd.DataFrame,
    rankings_df: pd.DataFrame,
    players_lookup,
) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for row in matches.itertuples(index=False, name="MatchRow"):
        record = canonicalize_ab(row._asdict(), rankings_df, players_lookup)
        for meta_key in ["tourney_id", "match_num", "score", "winner_name", "loser_name"]:
            if hasattr(row, meta_key):
                record[meta_key] = getattr(row, meta_key)
        records.append(record)
    df = pd.DataFrame(records)
    df = add_one_hot_features(df)
    df = add_recent_form_features(history_matches, df)
    df = add_derived_features(df)
    return df


def _align_features(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    for col in feature_cols:
        if col not in df.columns:
            df[col] = 0.0
    return df[feature_cols]


def _accuracy(y_true: np.ndarray, prob_a: np.ndarray) -> float:
    preds = (prob_a >= 0.5).astype(int)
    return float((preds == y_true).mean()) if len(y_true) else 0.0


def _evaluate_predictions(y_true: np.ndarray, prob_a: np.ndarray) -> dict[str, Any]:
    if len(y_true) == 0:
        return {"n": 0, "auc": None, "logloss": None, "brier": None, "accuracy": None}
    auc = roc_auc_score(y_true, prob_a) if len(np.unique(y_true)) > 1 else None
    logloss = log_loss(y_true, prob_a, labels=[0, 1])
    brier = brier_score_loss(y_true, prob_a)
    accuracy = _accuracy(y_true, prob_a)
    return {
        "n": int(len(y_true)),
        "auc": float(auc) if auc is not None else None,
        "logloss": float(logloss),
        "brier": float(brier),
        "accuracy": float(accuracy),
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    data_root = Path(args.data_root)

    try:
        bundle = joblib.load(args.model_path)
    except Exception as exc:
        print(f"Erreur: impossible de charger le modele: {args.model_path} ({exc})", file=sys.stderr)
        return 2

    model = bundle.get("model")
    feature_cols = bundle.get("features", [])
    if model is None or not feature_cols:
        print("Erreur: modele invalide (champs 'model' ou 'features' manquants).", file=sys.stderr)
        return 2

    matches_path = args.matches
    if matches_path is None:
        matches_path = data_root / "atp_matches" / f"atp_matches_{args.year}.csv"

    try:
        tournament_matches = _load_matches_file(matches_path)
    except FileNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        print("Astuce: fournissez --matches avec le fichier Wimbledon 2025.", file=sys.stderr)
        return 2

    tournament_matches = _filter_tournament(
        tournament_matches,
        tourney_name=args.tourney_name,
        year=args.year,
        tourney_level=args.tourney_level or None,
    )
    if tournament_matches.empty:
        print("Aucun match trouve pour ce filtre (tournoi/annee).", file=sys.stderr)
        return 2

    history_matches = _load_history_matches(data_root, args.year, args.history_years)
    history_matches = _merge_history(history_matches, tournament_matches)

    hub = DataHub(data_root)
    players_df, lookup = prepare_players(hub.load_players())
    rankings_df = prepare_rankings(hub.load_rankings())
    _ = players_df  # keep for side-effects/consistency

    dataset = _build_eval_dataset(tournament_matches, history_matches, rankings_df, lookup)
    if dataset.empty:
        print("Dataset vide apres construction.", file=sys.stderr)
        return 2

    X = _align_features(dataset, feature_cols)
    y = dataset["y"].astype(int).to_numpy()
    prob_a = model.predict_proba(X)[:, 1].astype(float)
    metrics = _evaluate_predictions(y, prob_a)

    print(f"Evaluation {args.tourney_name} {args.year}")
    print(f"Matches: {metrics['n']}")
    print(f"Accuracy: {metrics['accuracy']:.3f}" if metrics["accuracy"] is not None else "Accuracy: n/a")
    print(f"AUC: {metrics['auc']:.3f}" if metrics["auc"] is not None else "AUC: n/a")
    print(f"LogLoss: {metrics['logloss']:.4f}" if metrics["logloss"] is not None else "LogLoss: n/a")
    print(f"Brier: {metrics['brier']:.4f}" if metrics["brier"] is not None else "Brier: n/a")

    args.report_out.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "tourney_name": args.tourney_name,
        "year": args.year,
        "tourney_level": args.tourney_level,
        "model_path": args.model_path,
        "feature_count": len(feature_cols),
        "metrics": metrics,
        "created_at": datetime.now(UTC).isoformat(),
    }
    with args.report_out.open("w", encoding="utf-8") as fh:
        import json
        json.dump(report, fh, indent=2)

    if args.predictions_out:
        preds = dataset.copy()
        preds["p_A_win"] = prob_a
        preds["p_B_win"] = 1.0 - prob_a
        preds["predicted_winner"] = np.where(preds["p_A_win"] >= 0.5, preds["A_name"], preds["B_name"])
        preds["actual_winner"] = np.where(preds["y"] == 1, preds["A_name"], preds["B_name"])
        preds["correct"] = (preds["predicted_winner"] == preds["actual_winner"]).astype(int)
        preds_cols = [
            "tourney_date",
            "round",
            "surface",
            "A_name",
            "B_name",
            "actual_winner",
            "predicted_winner",
            "p_A_win",
            "p_B_win",
            "correct",
            "score",
            "tourney_id",
            "match_num",
        ]
        existing = [c for c in preds_cols if c in preds.columns]
        args.predictions_out.parent.mkdir(parents=True, exist_ok=True)
        preds[existing].to_csv(args.predictions_out, index=False)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
