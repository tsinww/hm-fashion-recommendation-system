import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hm_recsys.data import SnapshotFeatures
from hm_recsys.inactive_pipeline import (
    InactiveUserState,
    _inactive_extra_features,
    _neighbor_purchase_candidates,
    _user_state,
)
from hm_recsys.inactive_dlrm import _pairwise_positions
from hm_recsys.candidates import CandidateUnion


class InactivePipelineTests(unittest.TestCase):
    def setUp(self):
        self.features = SnapshotFeatures(
            customer_ids=np.array(["u"]),
            article_ids=np.array(["a", "b"]),
            user_cats=np.zeros((1, 1), dtype=np.int64),
            user_dense=np.zeros((1, 1), dtype=np.float32),
            item_cats=np.zeros((2, 1), dtype=np.int64),
            item_dense=np.zeros((2, 1), dtype=np.float32),
            customer_to_row={"u": 0},
            article_to_row={"a": 0, "b": 1},
            purchased={},
            item_popularity=np.array([2, 1], dtype=np.float32),
        )

    def test_days_since_purchase_is_relative_to_prediction_cutoff(self):
        history = pd.DataFrame(
            {
                "customer_id": ["u"],
                "article_id": ["a"],
                "t_dat": pd.to_datetime(["2020-08-01"]),
            }
        )
        state, _ = _user_state(
            ["u"], history, "2020-08-19", "2020-09-16", 180, self.features
        )
        self.assertEqual(state.days_since_last_purchase[0], 46)

    def test_missing_history_has_explicit_indicator(self):
        state = InactiveUserState(
            users=["u"],
            has_history_180d=np.array([0], dtype=np.float32),
            has_any_history=np.array([0], dtype=np.float32),
            days_since_last_purchase=np.array([np.nan], dtype=np.float32),
            historical_purchase_count=np.array([0], dtype=np.float32),
            historical_seen_current={},
        )
        values = _inactive_extra_features(state, np.array([0]))
        self.assertEqual(values[0, 4], 1.0)
        self.assertFalse(np.isnan(values).any())

    def test_behavioral_knn_does_not_fake_candidates_without_profile(self):
        rows = _neighbor_purchase_candidates(
            query_profiles=np.zeros((1, 2), dtype=np.float32),
            reference_profiles=np.array([[1.0, 0.0]], dtype=np.float32),
            reference_users=["v"],
            recent_items=[[0]],
            features=self.features,
            neighbors=1,
            k=2,
        )
        self.assertTrue((rows == -1).all())

    def test_pairwise_dlrm_negatives_come_from_same_customer_pool(self):
        union = CandidateUnion(
            items=np.array([[0, 1, 2], [1, 2, 0]], dtype=np.int64),
            source_rr=np.zeros((2, 3, 1), dtype=np.float32),
            source_names=("source",),
        )
        users, positives, negatives, stats = _pairwise_positions(
            ["u", "v"],
            union,
            np.array(["a", "b", "c"]),
            {"u": {"a"}, "v": {"c"}},
            negatives_per_positive=2,
            seed=42,
        )
        for user, positive, negative in zip(users, positives, negatives, strict=True):
            self.assertIn(positive, range(3))
            self.assertIn(negative, range(3))
            self.assertNotEqual(
                union.items[user, positive], union.items[user, negative]
            )
        self.assertEqual(stats["training_pairs"], 4)


if __name__ == "__main__":
    unittest.main()
