import unittest
from unittest.mock import patch

from src.check_device import probe


class DeviceTests(unittest.TestCase):
    def test_probe_has_stable_schema_without_torch(self):
        with patch.dict("sys.modules", {"torch": None}):
            result = probe()
        self.assertEqual(set(result), {"platform", "python", "torch_available", "cuda", "mps", "recommended"})
        self.assertEqual(result["recommended"], "cpu")


if __name__ == "__main__":
    unittest.main()
