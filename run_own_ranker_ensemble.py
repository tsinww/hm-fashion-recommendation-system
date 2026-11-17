#!/usr/bin/env python3
"""Train LightGBM and CatBoost on the saved four-week project candidates."""

from __future__ import annotations

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
        environment = os.environ.copy()
        environment["DYLD_LIBRARY_PATH"] = str(openmp.parent)
        environment["HM_LGBM_REEXEC"] = "1"
        os.execve(sys.executable, [sys.executable, *sys.argv], environment)

from hm_recsys.own_ranker_ensemble import (  # noqa: E402
    OwnEnsembleConfig,
    run_own_ranker_ensemble,
)


if __name__ == "__main__":
    result = run_own_ranker_ensemble(ROOT, OwnEnsembleConfig())
    print(f"Results written to {result}")

