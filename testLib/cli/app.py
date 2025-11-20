"""Standalone CLI to predict match winners using testLib."""
from __future__ import annotations

import argparse
import warnings
from pathlib import Path
from typing import Sequence

from .. import (
    TrainConfig,
    latest_match_for_player,
    load_matches,
    predict_match,
    train_model,
)
from ..data.matches import resolve_data_root

warnings.filterwarnings("ignore")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="TieBreaker", description="Predict tennis outcomes using Decision Trees")
    subparsers = parser.add_subparsers(dest="command", required=True)

    match = subparsers.add_parser("match", help="Predict the winner between two players")
    match.add_argument("--p1", required=True, help="Player 1")
    match.add_argument("--p2", required=True, help="Player 2")
    match.add_argument("--data-root", type=Path, default=resolve_data_root(), help="Custom data directory")
    match.add_argument("--years", nargs="*", type=int, help="Explicit match years to consider")
    match.add_argument("--train", action="store_true", help="Retrain the model before predicting")
    match.add_argument("--max-depth", type=int, help="Max depth for DecisionTree during optional retraining")
    match.set_defaults(func=cmd_match)

    train = subparsers.add_parser("train", help="Train the DecisionTree model on recent data")
    train.add_argument("--data-root", type=Path, default=resolve_data_root())
    train.add_argument("--max-depth", type=int)
    train.add_argument("--years", type=int, default=6, help="Number of recent years to include")
    train.set_defaults(func=cmd_train)

    return parser


def _ensure_model(args) -> None:
    if args.train:
        cfg = TrainConfig(max_depth=args.max_depth, years=args.years)
        result = train_model(cfg, data_root=args.data_root)
        print(f"Accuracy: {result['accuracy']:.4f}")


def cmd_train(args) -> int:
    cfg = TrainConfig(max_depth=args.max_depth, years=args.years)
    metrics = train_model(cfg, data_root=args.data_root)
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    return 0


def cmd_match(args) -> int:
    _ensure_model(args)
    matches = load_matches(data_root=args.data_root, years=args.years)
    match_a = latest_match_for_player(matches, args.p1)
    match_b = latest_match_for_player(matches, args.p2)
    if match_a is None or match_b is None:
        missing = []
        if match_a is None:
            missing.append(args.p1)
        if match_b is None:
            missing.append(args.p2)
        raise SystemExit(f"No recent matches found for: {', '.join(missing)}")

    result = predict_match(match_a, match_b, args.p1, args.p2)
    print(f"Winner: {result['winner']}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
