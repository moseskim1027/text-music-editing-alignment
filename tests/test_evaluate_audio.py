import tempfile
import unittest
from pathlib import Path

import numpy as np
import soundfile as sf

from src.evaluate_audio import compare


class AudioEvaluationTests(unittest.TestCase):
    def test_report_contains_proxies_and_unavailable_preference(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = np.zeros(100, dtype="float32")
            target = np.ones(100, dtype="float32") * 0.5
            generated = target.copy()
            for name, audio in (("source.wav", source), ("target.wav", target), ("generated.wav", generated)):
                sf.write(root / name, audio, 16000)
            report = compare(root / "source.wav", root / "target.wav", root / "generated.wav")
            self.assertEqual(report["adherence_proxy"], 1.0)
            self.assertIsNone(report["preference_win_rate"])
