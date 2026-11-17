#!/usr/bin/env python3
"""Run practical reproductions inspired by the published first/second solutions."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

if sys.platform == "darwin" and os.environ.get("HM_LGBM_REEXEC") != "1":
    candidates = [
        ROOT / ".venv/lib/python3.12/site-packages/sklearn/.dylibs/libomp.dylib",
        ROOT / ".venv/lib/python3.12/site-packages/torch/lib/libomp.dylib",
        Path("/Library/Frameworks/R.framework/Versions/4.4-arm64/Resources/lib/libomp.dylib"),
    ]
    openmp = next((path for path in candidates if path.exists()), None)
    if openmp is not None:
        environment = os.environ.copy(); environment["DYLD_LIBRARY_PATH"] = str(openmp.parent); environment["HM_LGBM_REEXEC"] = "1"
        os.execve(sys.executable, [sys.executable, *sys.argv], environment)

from hm_recsys.top_solution_reproductions import (  # noqa: E402
    TopSolutionConfig,
    run_first_place_inspired,
    run_first_place_ensemble,
    run_second_place_inspired,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--solution", choices=("first", "first-ensemble", "second", "both"), default="both")
    args = parser.parse_args()
    config = TopSolutionConfig()
    if args.solution in ("first", "both"):
        print(f"First-place-inspired results: {run_first_place_inspired(ROOT, config)}")
    if args.solution == "first-ensemble":
        print(f"First-place ensemble results: {run_first_place_ensemble(ROOT, config)}")
    if args.solution in ("second", "both"):
        print(f"Second-place-inspired results: {run_second_place_inspired(ROOT, config)}")


if __name__ == "__main__":
    main()

