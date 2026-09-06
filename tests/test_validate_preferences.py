import json
import tempfile
import unittest
from pathlib import Path

from src.validate_preferences import load_preferences, summarize


def preference(preferred="a"):
    return {"example_id": "one", "instruction": "Remove vocals", "candidate_a": "a.wav", "candidate_b": "b.wav", "preferred": preferred, "criteria": ["requested_edit", "untouched_content"], "annotator_id": "anon"}


class PreferenceTests(unittest.TestCase):
    def write(self, records):
        handle = tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False)
        with handle:
            for record in records:
                handle.write(json.dumps(record) + "\n")
        return Path(handle.name)

    def test_summary(self):
        self.assertEqual(summarize(self.write([preference(), preference("b")])), {"records": 2, "by_preference": {"a": 1, "b": 1}})

    def test_rejects_same_candidates(self):
        item = preference()
        item["candidate_b"] = item["candidate_a"]
        with self.assertRaisesRegex(ValueError, "must differ"):
            load_preferences(self.write([item]))

    def test_rejects_invalid_preference(self):
        with self.assertRaisesRegex(ValueError, "preferred"):
            load_preferences(self.write([preference("unknown")]))


if __name__ == "__main__":
    unittest.main()
