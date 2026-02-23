import os
import sys
import tempfile
import unittest
import json
import types
from unittest import mock

# train_mock_galaxy imports simflower at module import time; provide a tiny
# stub so arg/config helpers can be tested in lightweight environments.
if "simflower" not in sys.modules:
    simflower_stub = types.ModuleType("simflower")
    simflower_stub.FlowMatchingTrainer = object
    sys.modules["simflower"] = simflower_stub

import train_mock_galaxy


class TestTrainMockGalaxyArgsAndCache(unittest.TestCase):
    def test_parse_args_allows_cache_without_data_path(self):
        argv = ["train_mock_galaxy.py", "--cache-path", "/tmp/cache.npz"]
        with mock.patch.object(sys, "argv", argv):
            args = train_mock_galaxy.parse_args()

        self.assertIsNone(args.data_path)
        self.assertEqual(args.cache_path, "/tmp/cache.npz")

    def test_parse_args_config_defaults_with_cli_override(self):
        with tempfile.TemporaryDirectory() as td:
            cfg_path = os.path.join(td, "train_cfg.json")
            with open(cfg_path, "w") as f:
                json.dump(
                    {
                        "cache_path": "/tmp/from_config_cache.npz",
                        "batch_size": 111,
                        "epochs": 3,
                    },
                    f,
                )

            argv = [
                "train_mock_galaxy.py",
                "--config", cfg_path,
                "--batch-size", "222",
            ]
            with mock.patch.object(sys, "argv", argv):
                args = train_mock_galaxy.parse_args()

        self.assertEqual(args.cache_path, "/tmp/from_config_cache.npz")
        self.assertEqual(args.batch_size, 222)  # CLI overrides config
        self.assertEqual(args.epochs, 3)

    def test_main_raises_when_no_cache_and_no_data_path(self):
        with tempfile.TemporaryDirectory() as td:
            missing_cache = os.path.join(td, "missing_cache.npz")
            argv = [
                "train_mock_galaxy.py",
                "--output-dir", td,
                "--cache-path", missing_cache,
                "--epochs", "1",
            ]
            with mock.patch.object(sys, "argv", argv):
                with self.assertRaisesRegex(ValueError, "No cache found and --data-path not provided"):
                    train_mock_galaxy.main()


if __name__ == "__main__":
    unittest.main()
