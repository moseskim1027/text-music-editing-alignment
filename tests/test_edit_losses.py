import unittest

import torch

from src.edit_losses import combined_loss, edit_loss, preservation_loss


class EditLossTests(unittest.TestCase):
    def test_perfect_predictions_have_zero_loss(self):
        target = torch.ones(4)
        self.assertEqual(float(edit_loss(target, target)), 0.0)
        self.assertEqual(float(preservation_loss(target, target)), 0.0)

    def test_combined_loss_exposes_tradeoff(self):
        values = combined_loss(torch.zeros(2), torch.ones(2), torch.ones(2), torch.zeros(2), preservation_weight=2.0)
        self.assertEqual(set(values), {"edit_loss", "preservation_loss", "total_loss"})
        self.assertAlmostEqual(float(values["total_loss"]), float(values["edit_loss"] + 2 * values["preservation_loss"]))


if __name__ == "__main__":
    unittest.main()
