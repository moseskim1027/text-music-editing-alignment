import json
import tempfile
import unittest
from pathlib import Path

from src.baseline_inference import plan_jobs


class BaselineInferenceTests(unittest.TestCase):
    def manifest(self):
        handle = tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False)
        with handle:
            handle.write(json.dumps({
                "example_id": "demo",
                "source_audio": "audio/source.wav",
                "instruction": "Remove the vocals",
                "operation": "remove",
                "target_stem": "vocals",
                "split": "test",
            }) + "\n")
        return Path(handle.name)

    def test_dry_run_plans_output(self):
        jobs = plan_jobs(self.manifest(), Path("outputs/baseline"), require_audio=False)
        self.assertEqual(jobs[0]["output_audio"], "outputs/baseline/demo.wav")

    def test_missing_audio_is_reported(self):
        with self.assertRaises(FileNotFoundError):
            plan_jobs(self.manifest(), Path("outputs/baseline"))


if __name__ == "__main__":
    unittest.main()
