#!/usr/bin/env python3
"""Measure how longer lookbacks convert four-week cold users into dormant users."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--predictions",
        type=Path,
        default=ROOT
        / "artifacts/predictions/hybrid_expanded_repeat_experiment_lightgbm_top12.csv",
    )
    parser.add_argument("--context-start", default="2020-08-19")
    parser.add_argument("--cutoff", default="2020-09-16")
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/cold_start_audit/results.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    users = set(
        pd.read_csv(args.predictions, dtype={"customer_id": str})["customer_id"].astype(str)
    )
    recent = pd.read_pickle(
        ROOT / "artifacts/cache/transactions_2020-08-12_to_2020-09-22.pkl"
    )
    recent["customer_id"] = recent["customer_id"].astype(str)
    context = recent[
        (recent["t_dat"] >= args.context_start) & (recent["t_dat"] < args.cutoff)
    ]
    history_users = set(context["customer_id"]) & users
    cold_users = users - history_users

    last_purchase: dict[str, str] = {}
    for chunk in pd.read_csv(
        ROOT / "Data/transactions_train.csv",
        usecols=["t_dat", "customer_id"],
        dtype=str,
        chunksize=2_000_000,
    ):
        rows = chunk[
            (chunk["t_dat"] < args.context_start)
            & chunk["customer_id"].isin(cold_users)
        ]
        for customer, date in rows.groupby("customer_id")["t_dat"].max().items():
            if date > last_purchase.get(customer, ""):
                last_purchase[customer] = date

    thresholds = {
        "90_days": "2020-05-21",
        "180_days": "2020-02-21",
        "365_days": "2019-08-20",
        "ever": "0000-00-00",
    }
    recovery = {}
    for name, start in thresholds.items():
        recovered = sum(last_purchase.get(user, "") >= start for user in cold_users)
        recovery[name] = {
            "recovered": recovered,
            "remaining": len(cold_users) - recovered,
            "recovered_fraction": recovered / len(cold_users),
        }

    buckets = {
        "29_to_90_days": sum(
            "2020-05-21" <= last_purchase.get(user, "") < "2020-08-19"
            for user in cold_users
        ),
        "91_to_180_days": sum(
            "2020-02-21" <= last_purchase.get(user, "") < "2020-05-21"
            for user in cold_users
        ),
        "181_to_365_days": sum(
            "2019-08-20" <= last_purchase.get(user, "") < "2020-02-21"
            for user in cold_users
        ),
        "over_365_days": sum(
            "0000-00-00" <= last_purchase.get(user, "") < "2019-08-20"
            for user in cold_users
        ),
        "no_prior_transaction": sum(user not in last_purchase for user in cold_users),
    }
    result = {
        "evaluation_customers": len(users),
        "four_week_history_customers": len(history_users),
        "four_week_cold_customers": len(cold_users),
        "lookback_recovery": recovery,
        "last_activity_buckets": buckets,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(f"Results written to {args.output}")


if __name__ == "__main__":
    main()
