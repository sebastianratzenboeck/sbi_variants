import unittest
import tempfile
import types
import sys

import numpy as np
import pandas as pd

from columns import ALL_VALUE_COLS, OBS_COLS, OBS_ERR_COLS, N_INTRINSIC, N_TRUE_MAG
from inference_utils import NormStats

# train_mock_galaxy imports simflower at module import time; provide a tiny
# stub so preprocessing helpers can be imported in lightweight test envs.
if "simflower" not in sys.modules:
    simflower_stub = types.ModuleType("simflower")
    simflower_stub.FlowMatchingTrainer = object
    sys.modules["simflower"] = simflower_stub

from train_mock_galaxy import OBS_ERROR_FLOOR, build_arrays, _compute_test_split_with_cluster_holdout
from value_transforms import (
    apply_forward_value_transforms_numpy,
    apply_inverse_value_transforms_numpy,
    default_value_transform_metadata,
)


class TestValueTransforms(unittest.TestCase):
    def test_forward_inverse_preserves_support(self):
        cols = ["rad", "Av", "logAge"]
        names, params = default_value_transform_metadata(cols)

        x = np.array(
            [
                [-3.0, -0.2, 8.0],
                [0.0, 0.0, 9.0],
                [2.5, 1.5, 10.0],
            ],
            dtype=np.float32,
        )

        y = apply_forward_value_transforms_numpy(x, names, params)
        x_inv = apply_inverse_value_transforms_numpy(y, names, params)

        self.assertTrue(np.all(x_inv[:, 0] >= 0.0))  # rad
        self.assertTrue(np.all(x_inv[:, 1] >= 0.0))  # Av
        np.testing.assert_allclose(x_inv[:, 2], x[:, 2], rtol=1e-6, atol=1e-6)  # unchanged col
        self.assertAlmostEqual(float(x_inv[0, 0]), 0.0, places=6)
        self.assertAlmostEqual(float(x_inv[0, 1]), 0.0, places=6)


class TestBuildArraysConstraints(unittest.TestCase):
    def _make_df(self):
        n = 2
        d = {}
        for c in ALL_VALUE_COLS:
            d[c] = np.zeros(n, dtype=np.float32)

        # Ensure observed block is observed for both rows.
        for c in OBS_COLS:
            d[c] = np.array([1.0, 1.1], dtype=np.float32)

        # Default all observed errors to a valid positive value.
        for c in OBS_ERR_COLS:
            d[c] = np.full(n, 0.02, dtype=np.float32)

        # Explicit stress cases.
        d["parallax_err"] = np.array([0.0, 0.05], dtype=np.float32)
        d["rad"] = np.array([-2.0, 2.0], dtype=np.float32)
        d["Av"] = np.array([-0.1, 0.5], dtype=np.float32)
        return pd.DataFrame(d)

    def test_build_arrays_rad_av_transform_and_parallax_floor(self):
        df = self._make_df()
        (
            values_norm,
            errors_norm,
            observed_mask,
            means,
            stds,
            value_transform_names,
            value_transform_params,
            _cluster_ids,
            _log_err_mean,
            _log_err_std,
        ) = build_arrays(df)

        # Means are computed in transformed space.
        rad_idx = ALL_VALUE_COLS.index("rad")
        av_idx = ALL_VALUE_COLS.index("Av")
        expected_rad = np.log(np.clip(df["rad"].values.astype(np.float32), 0.0, None) + 1e-6)
        expected_av = np.log1p(np.clip(df["Av"].values.astype(np.float32), 0.0, None))
        self.assertAlmostEqual(float(means[rad_idx]), float(expected_rad.mean()), places=6)
        self.assertAlmostEqual(float(means[av_idx]), float(expected_av.mean()), places=6)

        # Use NormStats denormalization path to verify physical support.
        with tempfile.TemporaryDirectory() as td:
            norm_path = f"{td}/norm_stats.npz"
            np.savez(
                norm_path,
                means=means.astype(np.float32),
                stds=stds.astype(np.float32),
                columns=np.array(ALL_VALUE_COLS, dtype=object),
                value_transform_names=np.asarray(value_transform_names, dtype=object),
                value_transform_params=np.asarray(value_transform_params, dtype=np.float32),
                log_err_mean=np.array(0.0, dtype=np.float32),
                log_err_std=np.array(1.0, dtype=np.float32),
            )
            norm_stats = NormStats(norm_path)

        values_phys = norm_stats.denormalize_numpy(values_norm)
        self.assertTrue(np.all(values_phys[:, rad_idx] >= 0.0))
        self.assertTrue(np.all(values_phys[:, av_idx] >= 0.0))

        # parallax_err(0.0) should be floored and log-scaled as a real error.
        parallax_obs_i = OBS_COLS.index("parallax_obs")
        parallax_node_idx = N_INTRINSIC + N_TRUE_MAG + parallax_obs_i
        self.assertNotAlmostEqual(float(errors_norm[0, parallax_node_idx]), -5.0, places=7)
        self.assertNotAlmostEqual(float(errors_norm[0, parallax_node_idx]), 5.0, places=7)

        # Reconstruct expected z-score for floored parallax_err from the observed-error block.
        err_obs = np.full((2, len(OBS_ERR_COLS)), 0.02, dtype=np.float32)
        perr_idx = OBS_ERR_COLS.index("parallax_err")
        err_obs[0, perr_idx] = OBS_ERROR_FLOOR
        err_obs[1, perr_idx] = 0.05
        logs = np.log(err_obs.reshape(-1))
        m = logs.mean()
        s = logs.std()
        if s < 1e-10:
            s = 1.0
        expected_z = (np.log(OBS_ERROR_FLOOR) - m) / s
        self.assertAlmostEqual(float(errors_norm[0, parallax_node_idx]), float(expected_z), places=6)

        self.assertEqual(observed_mask.shape, values_norm.shape)


class TestClusterAwareSplit(unittest.TestCase):
    def test_cluster_holdout_excludes_whole_clusters_from_trainval(self):
        # cluster IDs: -1 field, positive=cluster labels
        cluster_ids = np.array([-1, -1, 1, 1, 2, 2, 3, 3, 4, 4], dtype=np.int64)
        trainval_idx, test_idx, heldout = _compute_test_split_with_cluster_holdout(
            n_total=len(cluster_ids),
            test_split=0.2,
            cluster_ids=cluster_ids,
            test_cluster_frac=0.5,  # hold out 2 of 4 clusters
            random_state=7,
        )
        self.assertGreaterEqual(len(heldout), 1)
        heldout_mask = np.isin(cluster_ids, heldout)
        # No held-out cluster star can be in train/val.
        self.assertFalse(np.any(heldout_mask[trainval_idx]))
        # All held-out cluster stars must be in test.
        self.assertTrue(np.all(np.isin(np.where(heldout_mask)[0], test_idx)))


if __name__ == "__main__":
    unittest.main()
