import json
import tempfile
import unittest
from pathlib import Path

from src.validate_manifest import summarize


def record(example_id="one", operation="remove", split="test"):
    return {
        "example_id": example_id,
        "source_audio": "source.wav",
        "instruction": "Remove the vocals",
        "operation": operation,
        "target_stem": "vocals",
        "split": split,
    }


class ManifestTests(unittest.TestCase):
    def write(self, rows):
        handle = tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False)
        with handle:
            for row in rows:
                handle.write(json.dumps(row) + "\n")
        return Path(handle.name)

    def test_summary_counts_records(self):
        second = record("two", "add", "train")
        second["source_audio"] = "other-source.wav"
        path = self.write([record("one"), second])
        self.assertEqual(summarize(path), {
            "records": 2,
            "by_operation": {"add": 1, "remove": 1},
            "by_split": {"test": 1, "train": 1},
        })

    def test_rejects_duplicate_ids(self):
        with self.assertRaisesRegex(ValueError, "duplicate"):
            summarize(self.write([record(), record()]))

    def test_rejects_unknown_operation(self):
        with self.assertRaisesRegex(ValueError, "unsupported operation"):
            summarize(self.write([record(operation="mix")]))

    def test_rejects_source_split_leakage(self):
        with self.assertRaisesRegex(ValueError, "multiple splits"):
            summarize(self.write([record("one", split="train"), record("two", "add", "test") | {"source_audio": "source.wav"}]))


if __name__ == "__main__":
    unittest.main()
