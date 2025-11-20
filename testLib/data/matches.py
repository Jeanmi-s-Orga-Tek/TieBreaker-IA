"""Raw match loading utilities for TieBreaker."""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Sequence

import pandas as pd

DEFAULT_DATA_ROOT = Path(__file__).resolve().parents[2] / "data"


def resolve_data_root(root: str | Path | None = None) -> Path:
    """Return the data root directory, defaulting to the repository's data folder."""
    if root is None:
        return DEFAULT_DATA_ROOT
    return Path(root).expanduser().resolve()


def _available_years(matches_dir: Path) -> list[int]:
    years = []
    for file in matches_dir.glob("atp_matches_*.csv"):
        suffix = file.stem.split("_")[-1]
        if suffix.isdigit():
            years.append(int(suffix))
    return sorted(set(years))


def _parse_match_date(value) -> pd.Timestamp | None:
    if pd.isna(value):
        return None
    try:
        return pd.Timestamp(datetime.strptime(str(int(value)), "%Y%m%d"))
    except Exception:
        try:
            return pd.to_datetime(value, errors="coerce")
        except Exception:
            return None


def load_matches(
    data_root: str | Path | None = None,
    years: Sequence[int] | None = None,
    limit_years: int = 6,
) -> pd.DataFrame:
    """Load ATP matches as a DataFrame filtered by years."""
    root = resolve_data_root(data_root)
    matches_dir = root / "atp_matches"
    if not matches_dir.exists():
        raise FileNotFoundError(f"Matches directory missing: {matches_dir}")

    if years:
        year_list = sorted({int(y) for y in years})
    else:
        available = _available_years(matches_dir)
        if limit_years and len(available) > limit_years:
            year_list = available[-limit_years:]
        else:
            year_list = available
    files = [matches_dir / f"atp_matches_{y}.csv" for y in year_list if (matches_dir / f"atp_matches_{y}.csv").exists()]
    if not files:
        raise FileNotFoundError("No match CSV files found for requested years.")

    frames = [pd.read_csv(f, low_memory=False) for f in sorted(files)]
    df = pd.concat(frames, ignore_index=True)

    if "tourney_date" in df.columns:
        df["tourney_date"] = df["tourney_date"].apply(_parse_match_date)
    else:
        df["tourney_date"] = pd.NaT

    for col in [
        "winner_name",
        "loser_name",
        "surface",
        "round",
        "score",
        "tourney_name",
    ]:
        if col in df.columns:
            df[col] = df[col].astype(str)
    for col in ["match_num", "minutes"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.sort_values(["tourney_date", "match_num"], na_position="last")
    return df.reset_index(drop=True)


def latest_match_for_player(matches: pd.DataFrame, player_name: str) -> pd.Series | None:
    """Return the most recent match row for the given player name."""
    normalized = player_name.casefold().strip()
    subset = matches[
        (matches["winner_name"].str.casefold().str.strip() == normalized)
        | (matches["loser_name"].str.casefold().str.strip() == normalized)
    ]
    if subset.empty:
        return None
    subset = subset.sort_values(["tourney_date", "match_num"], na_position="last")
    return subset.iloc[-1]
