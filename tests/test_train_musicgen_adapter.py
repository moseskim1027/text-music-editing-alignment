import unittest
from types import SimpleNamespace

from src.train_musicgen_adapter import normalize_decoder_start_token


class MusicGenAdapterTests(unittest.TestCase):
    def test_decoder_start_token_is_promoted(self):
        model = SimpleNamespace(config=SimpleNamespace(decoder=SimpleNamespace(decoder_start_token_id=2048)))
        normalize_decoder_start_token(model)
        self.assertEqual(model.config.decoder_start_token_id, 2048)

    def test_existing_decoder_start_token_is_preserved(self):
        model = SimpleNamespace(config=SimpleNamespace(decoder_start_token_id=7, decoder=SimpleNamespace(decoder_start_token_id=2048)))
        normalize_decoder_start_token(model)
        self.assertEqual(model.config.decoder_start_token_id, 7)
