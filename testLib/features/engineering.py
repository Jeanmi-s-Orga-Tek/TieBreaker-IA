"""Feature extraction logic."""
from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd

_PLAYER_FIELDS = {
    "ace": "ace",
    "df": "df",
    "svpt": "svpt",
    "1stIn": "1stIn",
    "1stWon": "1stWon",
    "2ndWon": "2ndWon",
    "bpSaved": "bpSaved",
    "bpFaced": "bpFaced",
    "rank": "rank",
    "rank_points": "rank_points",
}


def player_role_in_match(row: pd.Series, player_name: str) -> str:
    name = player_name.casefold().strip()
    if row["winner_name"].casefold().strip() == name:
        return "winner"
    if row["loser_name"].casefold().strip() == name:
        return "loser"
    raise ValueError(f"Player {player_name} not present in provided match row")


def extract_player_stats(row: pd.Series, role: str) -> Dict[str, float]:
    prefix = "winner_" if role == "winner" else "loser_"
    stats = {}
    for dst, src in _PLAYER_FIELDS.items():
        key = f"{prefix}{src}"
        value = row.get(key)
        stats[dst] = float(value) if pd.notna(value) else np.nan
    stats["games"] = float(row.get("w_SvGms" if role == "winner" else "l_SvGms", np.nan))
    stats["name"] = row["winner_name" if role == "winner" else "loser_name"].strip()

    svpt = stats.get("svpt", np.nan)
    if svpt and not np.isnan(svpt):
        stats["first_serve_pct"] = (stats.get("1stIn", 0.0) / svpt)
        stats["first_win_pct"] = (stats.get("1stWon", 0.0) / stats.get("1stIn", 1.0)) if stats.get("1stIn") else 0.0
        stats["second_win_pct"] = stats.get("2ndWon", 0.0) / max(svpt - stats.get("1stIn", 0.0), 1.0)
    else:
        stats["first_serve_pct"] = 0.0
        stats["first_win_pct"] = 0.0
        stats["second_win_pct"] = 0.0

    stats["bp_save_pct"] = stats.get("bpSaved", 0.0) / max(stats.get("bpFaced", 0.0), 1.0)
    stats["aggressiveness"] = stats.get("ace", 0.0) - stats.get("df", 0.0)
    stats["pressure_points"] = stats.get("bpFaced", 0.0)
    return stats


def _diff(a: float | int | np.ndarray, b: float | int | np.ndarray) -> float:
    if (a is None or np.isnan(a)) and (b is None or np.isnan(b)):
        return 0.0
    if a is None or np.isnan(a):
        return -float(b)
    if b is None or np.isnan(b):
        return float(a)
    return float(a) - float(b)


def build_pair_features(p1: Dict[str, float], p2: Dict[str, float]) -> Dict[str, float]:
    return {
        "rank_diff": _diff(p1.get("rank"), p2.get("rank")),
        "rank_points_diff": _diff(p1.get("rank_points"), p2.get("rank_points")),
        "ace_diff": _diff(p1.get("ace"), p2.get("ace")),
        "df_diff": _diff(p1.get("df"), p2.get("df")),
        "first_serve_pct_diff": _diff(p1.get("first_serve_pct"), p2.get("first_serve_pct")),
        "first_win_pct_diff": _diff(p1.get("first_win_pct"), p2.get("first_win_pct")),
        "second_win_pct_diff": _diff(p1.get("second_win_pct"), p2.get("second_win_pct")),
        "bp_save_pct_diff": _diff(p1.get("bp_save_pct"), p2.get("bp_save_pct")),
        "aggressiveness_diff": _diff(p1.get("aggressiveness"), p2.get("aggressiveness")),
        "pressure_points_diff": _diff(p1.get("pressure_points"), p2.get("pressure_points")),
        "svpt_diff": _diff(p1.get("svpt"), p2.get("svpt")),
        "p1_rank": p1.get("rank", np.nan),
        "p2_rank": p2.get("rank", np.nan),
        "p1_name": p1.get("name"),
        "p2_name": p2.get("name"),
    }


def feature_vector_from_matches(
    match_a: pd.Series,
    match_b: pd.Series,
    player_a: str,
    player_b: str,
) -> Dict[str, float]:
    role_a = player_role_in_match(match_a, player_a)
    role_b = player_role_in_match(match_b, player_b)
    stats_a = extract_player_stats(match_a, role_a)
    stats_b = extract_player_stats(match_b, role_b)
    return build_pair_features(stats_a, stats_b)

