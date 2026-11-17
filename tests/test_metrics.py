import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hm_recsys.metrics import average_precision_at_k, recall_at_k


class MetricTests(unittest.TestCase):
    def test_recall(self):
        self.assertEqual(recall_at_k(["a", "b", "c"], {"a", "c"}, 2), 0.5)

    def test_average_precision(self):
        # Relevant hits at ranks 1 and 3: (1/1 + 2/3) / 2.
        self.assertAlmostEqual(
            average_precision_at_k(["a", "x", "b"], {"a", "b"}, 3), 5 / 6
        )

    def test_duplicate_recommendation_counts_once(self):
        self.assertEqual(average_precision_at_k(["a", "a"], {"a"}, 2), 1.0)


if __name__ == "__main__":
    unittest.main()

