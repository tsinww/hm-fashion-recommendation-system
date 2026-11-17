#!/usr/bin/env python3
"""Rank the saved inactive-user pool with a pairwise DLRM."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from hm_recsys.inactive_dlrm import InactiveDLRMConfig, run_inactive_dlrm  # noqa: E402


if __name__ == "__main__":
    result = run_inactive_dlrm(ROOT, InactiveDLRMConfig())
    print(f"Results written to {result}")

