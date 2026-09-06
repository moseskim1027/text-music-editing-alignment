import tempfile
import unittest
from pathlib import Path

from src.prepare_slakh_manifest import build_records


class SlakhManifestTests(unittest.TestCase):
    def test_builds_add_and_remove_records(self):
        with tempfile.TemporaryDirectory() as directory:
            track = Path(directory) / "train" / "Track00001"
            (track / "stems").mkdir(parents=True)
            (track / "mix.flac").touch()
            (track / "stems" / "Bass.flac").touch()
            records = build_records(Path(directory))
            self.assertEqual(len(records), 2)
            self.assertEqual({r["operation"] for r in records}, {"add", "remove"})
            self.assertTrue(all(r["split"] == "train" for r in records))


if __name__ == "__main__":
    unittest.main()
