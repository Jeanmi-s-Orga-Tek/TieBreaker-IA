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
    return ap

def main(argv=None):
    argv = argv or sys.argv[1:]
    ap = build_parser()
    args = ap.parse_args(argv)
    hub = DataHub(args.data_root)
    return args.func(args, hub)

if __name__ == "__main__":
    raise SystemExit(main())
