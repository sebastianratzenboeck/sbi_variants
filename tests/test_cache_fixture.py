import unittest

import numpy as np

from data import load_cache_arrays
from tests.data_paths import CACHE_FIXTURE_PATH, INDEX_FIXTURE_PATH


class TestCacheFixture(unittest.TestCase):
    def test_fixture_files_exist_and_are_consistent(self):
        self.assertTrue(CACHE_FIXTURE_PATH.exists(), f"missing cache fixture: {CACHE_FIXTURE_PATH}")
        self.assertTrue(INDEX_FIXTURE_PATH.exists(), f"missing index fixture: {INDEX_FIXTURE_PATH}")

        cache = load_cache_arrays(str(CACHE_FIXTURE_PATH))
        idx = np.load(INDEX_FIXTURE_PATH).astype(np.int64)

        self.assertEqual(cache.values_norm.shape, cache.errors_norm.shape)
        self.assertEqual(cache.values_norm.shape, cache.observed_mask.shape)
        self.assertGreater(cache.values_norm.shape[0], 0)
        self.assertTrue(np.array_equal(idx, np.arange(len(idx), dtype=np.int64)))
        self.assertLess(idx.max(initial=0), cache.values_norm.shape[0])


if __name__ == "__main__":
    unittest.main()
