import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hm_recsys.top_solution_reproductions import _rank_fusion


class TopSolutionReproductionTests(unittest.TestCase):
    def test_equal_rank_fusion_rewards_cross_model_consensus(self):
        first = np.array([[3.0, 2.0, 1.0]], dtype=np.float32)
        second = np.array([[1.0, 3.0, 2.0]], dtype=np.float32)
        fused = _rank_fusion(first, second)
        # Item 1 is ranked second/first and should beat first/third and third/second.
        self.assertEqual(int(np.argmax(fused[0])), 1)


if __name__ == "__main__":
    unittest.main()
