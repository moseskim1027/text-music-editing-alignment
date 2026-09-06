import tempfile
import unittest
from pathlib import Path

import numpy as np
import soundfile as sf

from src.build_edit_targets import render_target


class EditTargetTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        root = Path(self.directory.name)
        self.mix = root / "mix.wav"
        self.target = root / "target.wav"
        self.replacement = root / "replacement.wav"
        sf.write(self.mix, np.full(8, 0.5, dtype="float32"), 16000)
        sf.write(self.target, np.full(8, 0.25, dtype="float32"), 16000)
        sf.write(self.replacement, np.full(8, 0.1, dtype="float32"), 16000)

    def tearDown(self):
        self.directory.cleanup()

    def read(self, name):
        return sf.read(Path(self.directory.name) / name, dtype="float32")[0].squeeze()

    def test_remove_subtracts_target_stem(self):
        output = Path(self.directory.name) / "remove.wav"
        render_target(self.mix, self.target, output, "remove")
        np.testing.assert_allclose(self.read("remove.wav"), 0.25, atol=1e-4)

    def test_adds_target_stem(self):
        output = Path(self.directory.name) / "add.wav"
        render_target(self.mix, self.target, output, "add")
        np.testing.assert_allclose(self.read("add.wav"), 0.75, atol=1e-4)

    def test_replaces_target_stem(self):
        output = Path(self.directory.name) / "replace.wav"
        render_target(self.mix, self.target, output, "replace", self.replacement)
        np.testing.assert_allclose(self.read("replace.wav"), 0.35, atol=1e-4)

    def test_replace_requires_replacement(self):
        with self.assertRaisesRegex(ValueError, "replacement stem"):
            render_target(self.mix, self.target, Path(self.directory.name) / "x.wav", "replace")
