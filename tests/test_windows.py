import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hm_recsys.data import window_with_history
from hm_recsys.top_solution_reproductions import (
    SECOND_FINAL,
    SECOND_STAGE1,
    SECOND_STAGE2,
)


class WindowTests(unittest.TestCase):
    def test_four_week_window_matches_existing_split(self):
        window = window_with_history("2020-09-16", "2020-09-23", 28)
        self.assertEqual(window.context_start, "2020-08-19")

    def test_eight_week_window(self):
        window = window_with_history("2020-09-16", "2020-09-23", 56)
        self.assertEqual(window.context_start, "2020-07-22")

    def test_six_week_window(self):
        window = window_with_history("2020-09-16", "2020-09-23", 42)
        self.assertEqual(window.context_start, "2020-08-05")

    def test_non_positive_history_rejected(self):
        with self.assertRaises(ValueError):
            window_with_history("2020-09-16", "2020-09-23", 0)

    def test_second_place_stages_are_strictly_temporal(self):
        self.assertEqual(SECOND_STAGE1.label_end, SECOND_STAGE2.label_start)
        self.assertEqual(SECOND_STAGE2.label_end, SECOND_FINAL.label_start)


if __name__ == "__main__":
    unittest.main()
