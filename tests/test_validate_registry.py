import json
import tempfile
import unittest
from pathlib import Path

from src.validate_registry import validate


def item(asset_id="one"):
    return {"asset_id": asset_id, "path": "audio.wav", "source": "provider", "license": "research", "attribution": "artist", "sha256": "a" * 64, "consent_status": "documented"}


class RegistryTests(unittest.TestCase):
    def write(self, records):
        handle = tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False)
        with handle:
            for record in records:
                handle.write(json.dumps(record) + "\n")
        return Path(handle.name)

    def test_valid_record(self):
        self.assertEqual(len(validate(self.write([item()]))), 1)

    def test_rejects_duplicate_ids(self):
        with self.assertRaisesRegex(ValueError, "duplicate"):
            validate(self.write([item(), item()]))

    def test_rejects_bad_checksum(self):
        invalid = item()
        invalid["sha256"] = "not-a-hash"
        with self.assertRaisesRegex(ValueError, "sha256"):
            validate(self.write([invalid]))


if __name__ == "__main__":
    unittest.main()
