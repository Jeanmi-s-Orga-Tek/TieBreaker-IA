##
## PROJECT PRO, 2025
## TieBreaker
## File description:
## tiebreaker_cli
##

from .models import DataHub
from testLib import (
    TrainConfig,
    latest_match_for_player,
    load_matches,
    predict_match,
    train_model,
)
from testLib.data.matches import resolve_data_root
from .predict_outcome import PredictRequest, predict_outcome
import argparse
import sys
import re
from pathlib import Path
from datetime import datetime
import pandas as pd
from difflib import get_close_matches

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
    return ap

def main(argv=None):
    argv = argv or sys.argv[1:]
    ap = build_parser()
    args = ap.parse_args(argv)
    hub = DataHub(args.data_root)
    return args.func(args, hub)

if __name__ == "__main__":
    raise SystemExit(main())
