"""Recent form feature utilities for TieBreaker."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import numpy as np
import pandas as pd


RECENT_FEATURE_MAP = {
    "win_rate": "win_rate_20_diff",
    "win_rate_surface": "win_rate_surface_20_diff",
    "first_in_pct": "first_in_pct_20_diff",
    "first_won_pct": "first_won_pct_20_diff",
    "second_won_pct": "second_won_pct_20_diff",
    "aces_per_SvGm": "aces_per_SvGm_20_diff",
    "df_per_SvGm": "df_per_SvGm_20_diff",
}


@dataclass(slots=True)
class RecentConfig:
    lookback_matches: int = 20
    min_matches: int = 5


def add_recent_form_features(
    matches: pd.DataFrame,
    dataset: pd.DataFrame,
    lookback_matches: int = 20,
    min_matches: int = 5,
) -> pd.DataFrame:
    """Augment dataset with rolling recent-form differentials."""

    cfg = RecentConfig(lookback_matches=lookback_matches, min_matches=min_matches)
    output = dataset.copy()
    if output.empty:
        return output

    history = _build_player_history(matches)
    history_lookup = _group_history_by_player(history)

    for col in ("recent_form_missing_A", "recent_form_missing_B"):
        if col not in output.columns:
            output[col] = 1
    for feat_col in RECENT_FEATURE_MAP.values():
        if feat_col not in output.columns:
            output[feat_col] = np.nan

    for idx, row in output.iterrows():
        target_date = pd.to_datetime(row.get("tourney_date"), errors="coerce")
        surface = str(row.get("surface") or "").strip().title() or None
        stats_a = _compute_recent_stats(history_lookup.get(row.get("A_player_id")), target_date, surface, cfg)
        stats_b = _compute_recent_stats(history_lookup.get(row.get("B_player_id")), target_date, surface, cfg)

        output.at[idx, "recent_form_missing_A"] = stats_a["missing"]
        output.at[idx, "recent_form_missing_B"] = stats_b["missing"]

        for base, col in RECENT_FEATURE_MAP.items():
            val_a = stats_a.get(base)
            val_b = stats_b.get(base)
            output.at[idx, col] = _diff_or_nan(val_a, val_b)

    return output


def _build_player_history(matches: pd.DataFrame) -> pd.DataFrame:
    if matches is None or matches.empty:
        return pd.DataFrame(columns=["player_id", "date", "surface"] + list(RECENT_FEATURE_MAP.keys()))
    df = matches.copy()
    df["tourney_date"] = pd.to_datetime(df["tourney_date"], errors="coerce")
    df = df.dropna(subset=["tourney_date"])
    records: list[Dict[str, Any]] = []
    for record in df.to_dict("records"):
        date = pd.to_datetime(record.get("tourney_date"), errors="coerce")
        if pd.isna(date):
            continue
        surface = str(record.get("surface") or "").strip().title() or None
        records.extend(
            filter(
                None,
                [
                    _build_player_record(record, "winner", date, surface),
                    _build_player_record(record, "loser", date, surface),
                ],
            )
        )
    history = pd.DataFrame(records)
    if history.empty:
        return history
    history = history.sort_values(["player_id", "date"]).reset_index(drop=True)
    return history


def _build_player_record(record: Dict[str, Any], role: str, date: pd.Timestamp, surface: str | None) -> Dict[str, Any] | None:
    prefix = "w_" if role == "winner" else "l_"
    lowered = {k.lower(): v for k, v in record.items()}
    player_id = _safe_player_id(record.get(f"{role}_id"))
    if player_id is None:
        return None
    stats = {
        "aces_per_SvGm": _safe_ratio(lowered.get(f"{prefix}ace"), lowered.get(f"{prefix}svgms")),
        "df_per_SvGm": _safe_ratio(lowered.get(f"{prefix}df"), lowered.get(f"{prefix}svgms")),
        "first_in_pct": _safe_ratio(lowered.get(f"{prefix}1stin"), lowered.get(f"{prefix}svpt")),
        "first_won_pct": _safe_ratio(lowered.get(f"{prefix}1stwon"), lowered.get(f"{prefix}1stin")),
        "second_won_pct": _safe_ratio(lowered.get(f"{prefix}2ndwon"), _second_serve_attempts(lowered, prefix)),
    }
    return {
        "player_id": player_id,
        "date": date,
        "surface": surface,
        "is_win": 1 if role == "winner" else 0,
        **stats,
    }


def _second_serve_attempts(record: Dict[str, Any], prefix: str) -> float | None:
    svpt = record.get(f"{prefix}svpt")
    first_in = record.get(f"{prefix}1stin")
    if svpt is None or first_in is None:
        return None
    try:
        value = float(svpt) - float(first_in)
    except (TypeError, ValueError):
        return None
    return value if value > 0 else None


def _safe_player_id(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, float) and np.isnan(value):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        try:
            return int(float(str(value).strip()))
        except (TypeError, ValueError):
            return None


def _group_history_by_player(history: pd.DataFrame) -> Dict[int, pd.DataFrame]:
    if history.empty:
        return {}
    lookup: Dict[int, pd.DataFrame] = {}
    for pid, group in history.groupby("player_id"):
        lookup[int(pid)] = group.reset_index(drop=True)
    return lookup


def _compute_recent_stats(
    history: pd.DataFrame | None,
    target_date: pd.Timestamp | None,
    surface: str | None,
    cfg: RecentConfig,
) -> Dict[str, Any]:
    base = {name: np.nan for name in RECENT_FEATURE_MAP.keys()}
    base["missing"] = 1
    if history is None or target_date is None or pd.isna(target_date):
        return base
    subset = history[history["date"] < target_date]
    if subset.empty:
        return base
    subset = subset.tail(cfg.lookback_matches)
    result: Dict[str, Any] = {}
    for name in ["aces_per_SvGm", "df_per_SvGm", "first_in_pct", "first_won_pct", "second_won_pct"]:
        result[name] = subset[name].mean() if name in subset.columns else np.nan
    result["win_rate"] = subset["is_win"].mean()
    if surface:
        surface_subset = subset[subset["surface"] == surface]
        result["win_rate_surface"] = surface_subset["is_win"].mean() if not surface_subset.empty else np.nan
    else:
        result["win_rate_surface"] = np.nan
    result["missing"] = int(len(subset) < cfg.min_matches)
    return result


def _safe_ratio(num: Any, denom: Any) -> float:
    try:
        num_val = float(num)
        denom_val = float(denom)
    except (TypeError, ValueError):
        return np.nan
    if denom_val == 0:
        return np.nan
    return num_val / denom_val


def _diff_or_nan(val_a: Any, val_b: Any) -> float:
    if pd.isna(val_a) or pd.isna(val_b):
        return np.nan
    return float(val_a) - float(val_b)
