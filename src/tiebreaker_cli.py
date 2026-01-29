##
## PROJECT PRO, 2025
## TieBreaker
## File description:
## tiebreaker_cli
##

try:
    from .models import DataHub
    from .predict_outcome import PredictRequest, predict_outcome
except ImportError:  # pragma: no cover - allow running via src on sys.path
    from models import DataHub
    from predict_outcome import PredictRequest, predict_outcome

from testLib import (
    TrainConfig,
    latest_match_for_player,
    load_matches,
    predict_match,
    train_model,
)
from testLib.data.matches import resolve_data_root
import argparse
import sys
import re
from pathlib import Path
from datetime import datetime
import pandas as pd
from difflib import get_close_matches
import joblib

try:
    from comparisonModels import evaluate_binary_prob_model, bootstrap_ci_diff
except ModuleNotFoundError:
    evaluate_binary_prob_model = None
    bootstrap_ci_diff = None

def norm(s: str) -> str:
    return re.sub(r'\s+', ' ', s.strip().casefold())

def best_name_match(query: str, candidates: list[str]) -> str | None:
    q = norm(query)
    for c in candidates:
        if norm(c) == q:
            return c
    m = get_close_matches(query, candidates, n=1, cutoff=0.75)
    return m[0] if m else None

def date_parse_or_none(s: str | None):
    if not s:
        return None
    try:
        return datetime.fromisoformat(s).date()
    except Exception:
        return None

def resolve_player_id(hub: DataHub, name_query: str):
    players = hub.load_players()
    candidates = players["full_name"].astype(str).tolist()
    match = best_name_match(name_query, candidates)
    if not match:
        return None, None
    row = players[players["full_name"] == match].iloc[0]
    return (int(row["player_id"]) if pd.notna(row["player_id"]) else None), str(row["full_name"])

def cmd_rank(args, hub: DataHub):
    pid, resolved = resolve_player_id(hub, args.player)
    if pid is None:
        print(f"Joueur introuvable: {args.player}", file=sys.stderr)
        return 1
    rankings = hub.load_rankings()

    if "player_id" in rankings.columns:
        df = rankings[rankings["player_id"] == pid]
    else:
        if "player_name_raw" in rankings.columns:
            firstname = resolved.split(" ")[0]
            lastname = resolved.split(" ")[-1]
            variants = {f"{lastname}, {firstname}".strip(), f"{firstname} {lastname}".strip(), resolved}
            df = rankings[rankings["player_name_raw"].astype(str).isin(variants)]
        else:
            df = pd.DataFrame()

    if df.empty:
        print(f"Aucun ranking trouvé pour {resolved} (player_id={pid}).")
        return 0

    target_date = date_parse_or_none(args.date)
    if "ranking_date" in df.columns and df["ranking_date"].notna().any():
        df = df.dropna(subset=["ranking_date"]).sort_values("ranking_date")
        if target_date:
            df = df[df["ranking_date"] <= target_date]
            if df.empty:
                print(f"Aucun ranking pour {resolved} avant {args.date}.")
                return 0
        row = df.iloc[-1]
        date_str = row["ranking_date"].isoformat()
    else:
        row = df.iloc[-1]
        date_str = "(date inconnue)"

    rank = int(row["rank"]) if "rank" in row and pd.notna(row["rank"]) else None
    points = int(row["points"]) if "points" in row and pd.notna(row["points"]) else None

    if rank is not None and points is not None:
        print(f"{resolved} — Rang ATP {rank} ({points} pts) au {date_str}")
    elif rank is not None:
        print(f"{resolved} — Rang ATP {rank} au {date_str}")
    else:
        print(f"Ranking introuvable pour {resolved} (au {date_str}).")
    return 0

def cmd_match(args, hub: DataHub):
    pid1, p1 = resolve_player_id(hub, args.p1)
    pid2, p2 = resolve_player_id(hub, args.p2)
    if pid1 is None or pid2 is None:
        if pid1 is None:
            print(f"Joueur P1 introuvable: {args.p1}", file=sys.stderr)
        if pid2 is None:
            print(f"Joueur P2 introuvable: {args.p2}", file=sys.stderr)
        return 1

    data_root = resolve_data_root(args.data_root)
    matches = load_matches(data_root=data_root, years=args.years)
    match_a = latest_match_for_player(matches, p1)
    match_b = latest_match_for_player(matches, p2)
    if match_a is None or match_b is None:
        missing = []
        if match_a is None:
            missing.append(p1)
        if match_b is None:
            missing.append(p2)
        print(f"Aucun historique récent pour: {', '.join(missing)}", file=sys.stderr)
        return 1

    def _predict():
        return predict_match(match_a, match_b, p1, p2)

    try:
        result = _predict()
    except FileNotFoundError:
        train_model(TrainConfig(years=args.years[0] if args.years else None), data_root=data_root)
        result = _predict()

    print(f"Winner: {result['winner']}")
    return 0

def _load_model_bundle_or_exit(path_value: str | Path):
    path = Path(path_value)
    if not path.exists():
        print(f"Erreur: modèle introuvable: {path}", file=sys.stderr)
        raise SystemExit(2)
    try:
        return joblib.load(path)
    except Exception as exc:
        print(f"Erreur: impossible de charger le modèle: {path} ({exc})", file=sys.stderr)
        raise SystemExit(2)

def _predict_proba_from_joblib_model(model_obj, X):
    """Return P(class=1) for a variety of sklearn-like estimators."""
    import numpy as np

    if hasattr(model_obj, "predict_proba"):
        proba = model_obj.predict_proba(X)
        # binary: columns [P(0), P(1)]
        return np.asarray(proba)[:, 1].astype(float)
    # fallback: hard predictions
    pred = model_obj.predict(X)
    return np.asarray(pred, dtype=float)


def _build_eval_xy(data_root: Path | None = None, years: int | None = 6):
    """Build an evaluation dataset compatible with testLib's DecisionTree.

    We reuse testLib's training sample generation so the feature set is numeric and stable.
    """
    from testLib.model.trainer import _build_training_samples, TrainConfig
    from testLib.data.matches import load_matches
    import pandas as pd
    import numpy as np

    cfg = TrainConfig(years=years)
    root = data_root
    matches = load_matches(root, limit_years=cfg.years or 6)
    records = _build_training_samples(matches)
    df = pd.DataFrame(records).dropna(axis=1, how="all")
    feature_df = df.select_dtypes(include=[np.number]).copy()
    y = feature_df.pop("target")
    return feature_df, y


# Place helpers near the top so they are easy to find.
def _format_model_load_error(model_path: str | Path, exc: Exception) -> str:
    """Return a user-friendly message for common pickle/joblib incompatibilities."""
    msg = str(exc)
    hint_lines = [
        f"Erreur lors de l'utilisation du modèle: {model_path}",
        f"Détail: {exc.__class__.__name__}: {msg}",
        "",
        "Causes fréquentes:",
        "- Incompatibilité de versions scikit-learn/xgboost entre l'entraînement et l'exécution (pickle/joblib).",
        "- Modèle sérialisé avec une autre version de sklearn.",
        "",
        "Solutions:",
        "- Ré-entraîner et re-sauvegarder le modèle dans cet environnement.",
        "- Ou installer la même version de scikit-learn/xgboost que celle utilisée à l'entraînement.",
    ]
    return "\n".join(hint_lines)


def cmd_comparison_models(args, hub: DataHub):
    _ = hub  # comparison currently doesn't need the DataHub
    if evaluate_binary_prob_model is None or bootstrap_ci_diff is None:
        print("Erreur: module `comparisonModels` introuvable.", file=sys.stderr)
        return 1

    # Accept both the old flag names (--model-a/--model-b) and the current ones (--m1/--m2)
    path_a = getattr(args, "model_a", None) or getattr(args, "m1")
    path_b = getattr(args, "model_b", None) or getattr(args, "m2")
    label_a = getattr(args, "label_a", None) or getattr(args, "l1", "model_a")
    label_b = getattr(args, "label_b", None) or getattr(args, "l2", "model_b")
    criterion = getattr(args, "criterion", "log_loss")

    model_a = _load_model_bundle_or_exit(path_a)
    model_b = _load_model_bundle_or_exit(path_b)

    # Build evaluation set (same for both models)
    # Note: this uses testLib's sample builder (numeric features aligned with testLib model).
    X, y = _build_eval_xy(data_root=args.data_root, years=6)

    # Some bundles may be dicts (e.g. {'model': clf, ...}); others are raw estimators.
    est_a = model_a.get("model") if isinstance(model_a, dict) and "model" in model_a else model_a
    est_b = model_b.get("model") if isinstance(model_b, dict) and "model" in model_b else model_b

    from sklearn.metrics import log_loss
    import numpy as np

    def _expected_feature_names(estimator):
        """Best-effort extraction of expected feature names for sklearn estimators/pipelines."""
        # Direct estimator
        names = getattr(estimator, "feature_names_in_", None)
        if names is not None:
            return list(names)

        # Pipelines: try final estimator
        if hasattr(estimator, "steps"):
            try:
                final_est = estimator.steps[-1][1]
                names = getattr(final_est, "feature_names_in_", None)
                if names is not None:
                    return list(names)
            except Exception:
                pass

            # Otherwise try any step
            for _, step in getattr(estimator, "steps", []):
                names = getattr(step, "feature_names_in_", None)
                if names is not None:
                    return list(names)

        return None

    def _align_X_for_estimator(estimator, X):
        expected = _expected_feature_names(estimator)
        if expected is None:
            return X
        X2 = X.reindex(columns=expected)
        return X2.fillna(0.0)

    try:
        p_a = _predict_proba_from_joblib_model(est_a, _align_X_for_estimator(est_a, X))
    except Exception as exc:
        print(_format_model_load_error(path_a, exc), file=sys.stderr)
        return 2

    try:
        p_b = _predict_proba_from_joblib_model(est_b, _align_X_for_estimator(est_b, X))
    except Exception as exc:
        print(_format_model_load_error(path_b, exc), file=sys.stderr)
        return 2

    m_a = evaluate_binary_prob_model(y, p_a)
    m_b = evaluate_binary_prob_model(y, p_b)

    # Bootstrap diff on logloss. Use labels=[0,1] to handle resamples with a single class.
    ci_lo, ci_hi, mean_diff = bootstrap_ci_diff(
        y,
        p_a,
        p_b,
        metric_fn=lambda yy, pp: log_loss(yy, np.clip(pp, 1e-15, 1 - 1e-15), labels=[0, 1]),
        n_boot=2000,
        seed=42,
    )

    # Decide winner according to criterion
    higher_is_better = {"accuracy", "auc"}
    if criterion not in {"log_loss", "brier", "accuracy", "auc"}:
        print(f"Critère inconnu: {criterion}. Choisissez parmi: log_loss, brier, accuracy, auc", file=sys.stderr)
        return 2

    a_val = float(m_a.get(criterion, float("nan")))
    b_val = float(m_b.get(criterion, float("nan")))
    if criterion in higher_is_better:
        winner = label_a if a_val > b_val else label_b
        direction = "plus grand"
    else:
        winner = label_a if a_val < b_val else label_b
        direction = "plus petit"

    report = {
        "criterion": criterion,
        "winner": winner,
        "model1": {"label": label_a, "path": str(path_a), "metrics": m_a},
        "model2": {"label": label_b, "path": str(path_b), "metrics": m_b},
        "log_loss_diff_mean": mean_diff,
        "log_loss_diff_ci95": [ci_lo, ci_hi],
        "n": int(len(y)),
    }

    # Save optional report
    if args.report_out:
        import json

        out = Path(args.report_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    # Clear, human-friendly output
    def _fmt(x):
        try:
            return f"{float(x):.4f}" if x == x else "nan"
        except Exception:
            return "nan"

    print("\n=== Model comparison ===")
    print(f"Dataset size: n={report['n']}")
    print(f"Chosen criterion: {criterion} ({direction} = meilleur)")
    print("\nMetrics:")
    print(f"  {label_a:>12} | log_loss={_fmt(m_a.get('log_loss'))} brier={_fmt(m_a.get('brier'))} acc={_fmt(m_a.get('accuracy'))} auc={_fmt(m_a.get('auc'))}")
    print(f"  {label_b:>12} | log_loss={_fmt(m_b.get('log_loss'))} brier={_fmt(m_b.get('brier'))} acc={_fmt(m_b.get('accuracy'))} auc={_fmt(m_b.get('auc'))}")

    print("\nVerdict:")
    print(f"  Winner: {winner} (meilleur {criterion}: {label_a}={_fmt(a_val)} vs {label_b}={_fmt(b_val)})")
    print("\nLog-loss robustness (bootstrap):")
    print(f"  diff (model1-model2) mean={_fmt(mean_diff)}  CI95=[{_fmt(ci_lo)}, { _fmt(ci_hi)}]")

    if args.report_out:
        print(f"\nReport written to: {args.report_out}")

    return 0


def cmd_predict(args, hub: DataHub):
    if args.date:
        try:
            date_value = pd.to_datetime(args.date).normalize()
        except Exception:
            print("--date doit être au format YYYY-MM-DD", file=sys.stderr)
            return 1
    else:
        date_value = None
    req = PredictRequest(
        p1_name=args.p1,
        p2_name=args.p2,
        date=date_value,
        surface=args.surface,
        round=args.round,
        best_of=args.best_of,
        data_root=str(args.data_root),
        model_path=str(args.model_path),
    )
    try:
        result = predict_outcome(req)
    except (ValueError, FileNotFoundError) as exc:
        print(f"Erreur: {exc}", file=sys.stderr)
        return 1

    meta = result.meta or {}
    date_display = meta.get("target_date")
    surface_display = meta.get("surface", args.surface or "Hard")
    round_display = meta.get("round", args.round or "R32")
    best_of_display = meta.get("best_of") or args.best_of

    print(f"Canonical A: {result.A_name}")
    print(f"Canonical B: {result.B_name}")
    timeline = f"Date: {date_display}" if date_display else "Date: (non spécifiée)"
    timeline += f"  Surface: {surface_display}  Round: {round_display}"
    if best_of_display:
        timeline += f"  Best-of-{best_of_display}"
    print(timeline)
    print(f"P(A gagne) = {result.p_A_win:.3f}")
    print(f"P(B gagne) = {result.p_B_win:.3f}")
    p1_label = meta.get("p1_resolved") or args.p1
    p2_label = meta.get("p2_resolved") or args.p2
    print(f"P(p1 gagne) [{p1_label}] = {result.p_p1_win:.3f}")
    print(f"P(p2 gagne) [{p2_label}] = {result.p_p2_win:.3f}")
    meta_bits = []
    if meta.get("model_type"):
        meta_bits.append(str(meta["model_type"]))
    if meta.get("train_end_year"):
        meta_bits.append(f"train<={meta['train_end_year']}")
    if meta.get("val_end_year"):
        meta_bits.append(f"val={meta['val_end_year']}")
    if meta_bits:
        print("Meta: " + ", ".join(meta_bits))
    return 0

def build_parser():
    ap = argparse.ArgumentParser(description="TieBreaker CLI — Parser ATP (rankings & matches)")
    ap.add_argument("--data-root", type=Path, default=Path("data"), help="Data root directory (default: ./data)")
    sp = ap.add_subparsers(dest="cmd", required=True)

    ap_rank = sp.add_parser("rank", help="Get a player's ATP ranking on a given date (or the most recent one)")
    ap_rank.add_argument("--player", required=True, help="Player name (ex: 'Novak Djokovic')")
    ap_rank.add_argument("--date", help="Date ISO (YYYY-MM-DD). If absent, take the last available ranking (current if available, otherwise historical).")
    ap_rank.set_defaults(func=cmd_rank)

    ap_match = sp.add_parser("match", help="Predict the winner between two players")
    ap_match.add_argument("--p1", required=True, help="Player 1")
    ap_match.add_argument("--p2", required=True, help="Player 2")
    ap_match.add_argument("--years", nargs="*", type=int, help="Optional list of years to draw recent matches from")
    ap_match.set_defaults(func=cmd_match)

    ap_predict = sp.add_parser("predict", help="Prédit l'issue d'un match entre deux joueurs")
    ap_predict.add_argument("--p1", required=True, help="Nom du premier joueur")
    ap_predict.add_argument("--p2", required=True, help="Nom du second joueur")
    ap_predict.add_argument("--date", help="Date du match (YYYY-MM-DD). Par défaut: fin de l'horizon du modèle")
    ap_predict.add_argument("--surface", default="Hard", help="Surface (Hard, Clay, Grass, Carpet)")
    ap_predict.add_argument("--round", default="R32", help="Round (F, SF, QF, R16, R32, ...)")
    ap_predict.add_argument("--best-of", type=int, help="Nombre de sets gagnants (3 ou 5). Si omis, inféré")
    ap_predict.add_argument("--model-path", default="models/outcome_model_xgb.pkl", help="Chemin vers le modèle entraîné")
    ap_predict.set_defaults(func=cmd_predict)

    ap_cmp = sp.add_parser("comparison", help="Comparez deux modèles formés")
    ap_cmp.add_argument("--m1", required=True, help="Chemin vers le premier bundle de modèles")
    ap_cmp.add_argument("--m2", required=True, help="Chemin vers le second bundle de modèles")
    ap_cmp.add_argument("--l1", "-l1", default="model_a", help="Nom d'affichage du modèle A")
    ap_cmp.add_argument("--l2", "-l2", default="model_b", help="Nom d'affichage du modèle B")
    ap_cmp.add_argument(
        "--criterion",
        choices=["log_loss", "brier", "accuracy", "auc"],
        default="log_loss",
        help="Critère principal pour décider du gagnant (défaut: log_loss)",
    )
    ap_cmp.add_argument("--report-out", help="Chemin optionnel pour rédiger un rapport de comparaison")
    ap_cmp.set_defaults(func=cmd_comparison_models)

    return ap

def main(argv=None):
    argv = argv or sys.argv[1:]
    ap = build_parser()
    args = ap.parse_args(argv)
    hub = DataHub(args.data_root)
    return args.func(args, hub)

if __name__ == "__main__":
    raise SystemExit(main())
