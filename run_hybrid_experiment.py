#!/usr/bin/env python3
"""Run the parallel multi-generator H&M recommendation experiment."""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

# The macOS LightGBM wheel needs libomp. This machine already has the runtime
# through R, but dyld must see that directory when Python starts. Re-exec once
# with the narrow library path instead of requiring a system-wide installation.
if sys.platform == "darwin" and os.environ.get("HM_LGBM_REEXEC") != "1":
    r_openmp = Path(
        "/Library/Frameworks/R.framework/Versions/4.4-arm64/Resources/lib/libomp.dylib"
    )
    if r_openmp.exists():
        environment = os.environ.copy()
        existing = environment.get("DYLD_LIBRARY_PATH", "")
        environment["DYLD_LIBRARY_PATH"] = (
            f"{r_openmp.parent}:{existing}" if existing else str(r_openmp.parent)
        )
        environment["HM_LGBM_REEXEC"] = "1"
        os.execve(sys.executable, [sys.executable, *sys.argv], environment)

from hm_recsys.hybrid_pipeline import (  # noqa: E402
    HYBRID_PROFILES,
    run_hybrid_experiment,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", choices=HYBRID_PROFILES, default="hybrid_smoke")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--max-train-users", type=int)
    parser.add_argument("--max-eval-users", type=int)
    parser.add_argument("--union-k", type=int)
    parser.add_argument(
        "--save-candidates",
        action="store_true",
        help="Persist train/evaluation candidate arrays for later reranking.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = HYBRID_PROFILES[args.profile]
    overrides = {
        key: value
        for key, value in {
            "seed": args.seed,
            "max_train_users": args.max_train_users,
            "max_eval_users": args.max_eval_users,
            "union_k": args.union_k,
        }.items()
        if value is not None
    }
    result = run_hybrid_experiment(
        ROOT, replace(config, **overrides), save_candidates=args.save_candidates
    )
    print(f"Results written to {result}")


if __name__ == "__main__":
    main()
