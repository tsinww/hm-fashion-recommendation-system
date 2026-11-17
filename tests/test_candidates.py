import unittest
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hm_recsys.candidates import CandidateSource, union_candidates
from hm_recsys.data import SnapshotFeatures


class CandidateUnionTests(unittest.TestCase):
    def setUp(self):
        self.features = SnapshotFeatures(
            customer_ids=np.array(["u"]),
            article_ids=np.array(["a", "b", "c", "d"]),
            user_cats=np.zeros((1, 1), dtype=np.int64),
            user_dense=np.zeros((1, 1), dtype=np.float32),
            item_cats=np.zeros((4, 1), dtype=np.int64),
            item_dense=np.zeros((4, 1), dtype=np.float32),
            customer_to_row={"u": 0},
            article_to_row={"a": 0, "b": 1, "c": 2, "d": 3},
            purchased={"u": {"a"}},
            item_popularity=np.array([4, 3, 2, 1], dtype=np.float32),
        )

    def test_discovery_filters_seen_but_repeat_restores_it(self):
        discovery = CandidateSource("bpr_mf", np.array([[0, 1, 2]]))
        repeat = CandidateSource("repeat", np.array([[0]]))
        result = union_candidates(
            ["u"],
            [discovery, repeat],
            self.features,
            max_candidates=3,
            remove_history_from={"bpr_mf"},
        )
        self.assertIn(0, result.items[0])
        repeat_position = int(np.flatnonzero(result.items[0] == 0)[0])
        self.assertEqual(result.source_rr[0, repeat_position, 0], 0.0)
        self.assertEqual(result.source_rr[0, repeat_position, 1], 1.0)
        self.assertEqual(len(set(result.items[0])), 3)

    def test_discovery_only_excludes_seen(self):
        discovery = CandidateSource("implicit_als", np.array([[0, 1, 2]]))
        result = union_candidates(
            ["u"],
            [discovery],
            self.features,
            max_candidates=3,
            remove_history_from={"implicit_als"},
        )
        self.assertNotIn(0, result.items[0])

    def test_direct_union_keeps_seen_items_without_repeat_source(self):
        discovery = CandidateSource("implicit_als", np.array([[0, 1, 2]]))
        result = union_candidates(
            ["u"],
            [discovery],
            self.features,
            max_candidates=3,
            remove_history_from=set(),
        )
        self.assertEqual(result.items[0].tolist(), [0, 1, 2])

    def test_allow_seen_union_retains_repeat_provenance(self):
        discovery = CandidateSource("implicit_als", np.array([[0, 1, 2]]))
        repeat = CandidateSource("repeat", np.array([[0]]))
        result = union_candidates(
            ["u"],
            [discovery, repeat],
            self.features,
            max_candidates=3,
            remove_history_from=set(),
        )
        seen_position = int(np.flatnonzero(result.items[0] == 0)[0])
        self.assertGreater(result.source_rr[0, seen_position, 0], 0.0)
        self.assertGreater(result.source_rr[0, seen_position, 1], 0.0)


if __name__ == "__main__":
    unittest.main()
