import unittest
from unittest.mock import patch
import json
import sys
import tempfile
from pathlib import Path

from src.log_evaluation import log_evaluation


class FakeRun:
    class Info:
        run_id = "test-run"
    info = Info()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


class FakeMLflow:
    def set_tracking_uri(self, value):
        self.uri = value

    def set_experiment(self, value):
        self.experiment = value

    def start_run(self):
        return FakeRun()

    def log_metrics(self, value):
        self.metrics = value

    def log_metric(self, key, value):
        self.single_metric = (key, value)

    def log_param(self, key, value):
        self.param = (key, value)

    def log_artifact(self, *args, **kwargs):
        self.artifact = args[0]


class LoggingTests(unittest.TestCase):
    def test_logs_summary_and_artifact(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as handle:
            handle.write(json.dumps({"example_id":"one","operation":"remove","edit_success":1,"target_change":.8,"preservation":.9,"alignment":.8,"preference":.7,"quality":.9}) + "\n")
            scores = Path(handle.name)
        fake = FakeMLflow()
        with patch.dict(sys.modules, {"mlflow": fake}):
            from src.log_evaluation import log_evaluation
            self.assertEqual(log_evaluation(scores, "test", "sqlite:///test.db"), "test-run")
        self.assertIn("mean_edit_success", fake.metrics)
        self.assertEqual(fake.artifact, str(scores))


if __name__ == "__main__":
    unittest.main()
