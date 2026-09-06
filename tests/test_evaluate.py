import unittest

from src.evaluate import aggregate


def score(example_id, operation, edit_success, preservation):
    return {
        "example_id": example_id,
        "operation": operation,
        "edit_success": edit_success,
        "target_change": 0.8,
        "preservation": preservation,
        "alignment": 0.7,
        "preference": 0.6,
        "quality": 0.9,
    }


class EvaluationTests(unittest.TestCase):
    def test_aggregates_means_and_adjusted_score(self):
        result = aggregate([score("a", "remove", 1.0, 0.8), score("b", "add", 0.5, 0.6)])
        self.assertEqual(result["records"], 2)
        self.assertAlmostEqual(result["mean"]["edit_success"], 0.75)
        self.assertAlmostEqual(result["preservation_adjusted_edit_success"], 0.525)
        self.assertEqual(set(result["by_operation"]), {"add", "remove"})

    def test_rejects_out_of_range_score(self):
        with self.assertRaisesRegex(ValueError, "between 0 and 1"):
            aggregate([score("a", "remove", 1.2, 0.8)])

    def test_rejects_empty_input(self):
        with self.assertRaisesRegex(ValueError, "at least one"):
            aggregate([])


if __name__ == "__main__":
    unittest.main()
