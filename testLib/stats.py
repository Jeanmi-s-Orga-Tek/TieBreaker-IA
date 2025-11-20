"""Lightweight helpers to compute descriptive statistics on match data."""
from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
from typing import Dict

import pandas as pd

__all__ = [
    "MatchStats",
    "compute_match_stats",
    "format_match_stats",
]


@dataclass
class MatchStats:
    total_matches: int
    unique_players: int
    date_range: tuple[str | None, str | None]
    surfaces: Dict[str, int]
    top_winners: Dict[str, int]

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


def _normalize_series(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.title()


def compute_match_stats(matches: pd.DataFrame, top_n: int = 5) -> MatchStats:
    if matches.empty:
        return MatchStats(0, 0, (None, None), {}, {})

    winners = _normalize_series(matches["winner_name"])
    losers = _normalize_series(matches["loser_name"])
    players = pd.unique(pd.concat([winners, losers], ignore_index=True))

    surfaces = _normalize_series(matches.get("surface", pd.Series(dtype=str)))
    surface_counts = (
        surfaces.value_counts().head(top_n).to_dict() if not surfaces.empty else {}
    )

    win_counter = Counter(winners)
    top_winners = dict(win_counter.most_common(top_n))

    dates = matches["tourney_date"].dropna()
    date_min = dates.min().date().isoformat() if not dates.empty else None
    date_max = dates.max().date().isoformat() if not dates.empty else None

    return MatchStats(
        total_matches=len(matches),
        unique_players=len(players),
        date_range=(date_min, date_max),
        surfaces=surface_counts,
        top_winners=top_winners,
    )


def format_match_stats(stats: MatchStats) -> str:
    """Return a readable multi-line summary for CLI/log output."""
    lines = [
        f"Total matches analysed: {stats.total_matches}",
        f"Unique players: {stats.unique_players}",
    ]

    start, end = stats.date_range
    if start or end:
        lines.append(f"Date range: {start or 'unknown'} → {end or 'unknown'}")

    if stats.surfaces:
        surf = ", ".join(f"{name}: {count}" for name, count in stats.surfaces.items())
        lines.append(f"Top surfaces: {surf}")

    if stats.top_winners:
        winners = ", ".join(f"{name}: {count}" for name, count in stats.top_winners.items())
        lines.append(f"Top winners: {winners}")

    return "\n".join(lines)
