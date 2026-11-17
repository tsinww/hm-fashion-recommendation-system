#!/usr/bin/env python3
"""Run a leakage-safe H&M recommendation experiment.

Examples:
    .venv/bin/python run_experiment.py --profile smoke
    .venv/bin/python run_experiment.py --profile experiment
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from hm_recsys.pipeline import PROFILES, run_experiment  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", choices=PROFILES, default="smoke")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--max-train-users", type=int)
    parser.add_argument("--max-eval-users", type=int)
    parser.add_argument("--candidate-k", type=int)
    parser.add_argument("--rerank-k", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = PROFILES[args.profile]
    overrides = {
        key: value
        for key, value in {
            "seed": args.seed,
            "max_train_users": args.max_train_users,
            "max_eval_users": args.max_eval_users,
            "candidate_k": args.candidate_k,
            "rerank_k": args.rerank_k,
        }.items()
        if value is not None
    }
    config = replace(config, **overrides)
    path = run_experiment(ROOT, config)
    print(f"Results written to {path}")


if __name__ == "__main__":
    main()

