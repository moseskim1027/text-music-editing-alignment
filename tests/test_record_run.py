import tempfile
import unittest
from pathlib import Path

from src.record_run import build_metadata


class RunMetadataTests(unittest.TestCase):
    def test_hashes_inputs_and_records_device(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config, manifest = root / "config.json", root / "manifest.jsonl"
            config.write_text('{"seed": 42}')
            manifest.write_text('{"example_id": "one"}\n')
            metadata = build_metadata(config, manifest, "mps")
            self.assertEqual(len(metadata["config_sha256"]), 64)
            self.assertEqual(len(metadata["manifest_sha256"]), 64)
            self.assertEqual(metadata["device"], "mps")


if __name__ == "__main__":
    unittest.main()
