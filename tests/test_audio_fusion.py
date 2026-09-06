import unittest

import torch

from src.audio_fusion import SourceAudioFusion


class AudioFusionTests(unittest.TestCase):
    def test_adds_one_audio_prefix_token(self):
        fusion = SourceAudioFusion(audio_dim=4, text_dim=6)
        output = fusion(torch.randn(2, 8, 4), torch.randn(2, 5, 6))
        self.assertEqual(tuple(output.shape), (2, 6, 6))

    def test_rejects_batch_mismatch(self):
        fusion = SourceAudioFusion(audio_dim=4, text_dim=6)
        with self.assertRaisesRegex(ValueError, "batch sizes"):
            fusion(torch.randn(2, 8, 4), torch.randn(1, 5, 6))
