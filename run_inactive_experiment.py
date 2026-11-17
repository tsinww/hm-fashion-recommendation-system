#!/usr/bin/env python3
"""Run the isolated four-week-inactive KNN retrieval and ranking experiment."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

if sys.platform == "darwin" and os.environ.get("HM_LGBM_REEXEC") != "1":
    openmp_candidates = [
        Path("/Library/Frameworks/R.framework/Versions/4.4-arm64/Resources/lib/libomp.dylib"),
        ROOT / ".venv/lib/python3.12/site-packages/sklearn/.dylibs/libomp.dylib",
        ROOT / ".venv/lib/python3.12/site-packages/torch/lib/libomp.dylib",
    ]
    openmp = next((path for path in openmp_candidates if path.exists()), None)
    if openmp is not None:
        environment = os.environ.copy()
        existing = environment.get("DYLD_LIBRARY_PATH", "")
        environment["DYLD_LIBRARY_PATH"] = (
            f"{openmp.parent}:{existing}" if existing else str(openmp.parent)
        )
        environment["HM_LGBM_REEXEC"] = "1"
        os.execve(sys.executable, [sys.executable, *sys.argv], environment)

from hm_recsys.inactive_pipeline import (  # noqa: E402
    INACTIVE_PROFILES,
    run_inactive_experiment,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--profile", choices=INACTIVE_PROFILES, default="inactive_knn_smoke"
    )
    parser.add_argument("--no-save-candidates", action="store_true")
    args = parser.parse_args()
    result = run_inactive_experiment(
        ROOT,
        INACTIVE_PROFILES[args.profile],
        save_candidates=not args.no_save_candidates,
    )
    print(f"Results written to {result}")


if __name__ == "__main__":
    main()
