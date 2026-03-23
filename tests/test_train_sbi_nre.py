import unittest

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from train_sbi_nre import _compute_test_split_with_cluster_holdout, _run_epoch


class _DictDataset(Dataset):
    def __init__(self):
        self.theta = torch.tensor([[0.0], [1.0]], dtype=torch.float32)
        self.inputs = torch.zeros((2, 1), dtype=torch.float32)
        self.errors = torch.zeros((2, 1), dtype=torch.float32)
        self.observed = torch.ones((2, 1), dtype=torch.float32)
        self.sample_weight = torch.tensor([2.0, 3.0], dtype=torch.float32)

    def __len__(self):
        return 2

    def __getitem__(self, idx):
        return {
            "theta": self.theta[idx],
            "inputs": self.inputs[idx],
            "errors": self.errors[idx],
            "observed": self.observed[idx],
            "sample_weight": self.sample_weight[idx],
        }


class _DummyRatioModel(torch.nn.Module):
    def encode_context(self, values, errors, observed_mask):
        return torch.zeros((values.shape[0], 1), dtype=values.dtype, device=values.device)

    def logits_from_context(self, theta, ctx, mask=None):
        return theta[:, 0]


class TestTrainSbiNreWeighting(unittest.TestCase):
    def test_negative_pairs_use_both_row_weights_under_curriculum_correction(self):
        loader = DataLoader(_DictDataset(), batch_size=2, shuffle=False, drop_last=False)
        model = _DummyRatioModel()

        stats = _run_epoch(
            model=model,
            loader=loader,
            device="cpu",
            train=False,
            optimizer=None,
            scaler=None,
            use_amp=False,
            grad_clip_norm=1.0,
            ratio_mask_mode="none",
            mask_bernoulli_p=0.5,
            use_balanced_loss=False,
            bnre_lambda=100.0,
            importance_mode="none",
            importance_beta=1.0,
            importance_min=0.1,
            importance_max=10.0,
            apply_importance=False,
        )

        logits_pos = torch.tensor([0.0, 1.0], dtype=torch.float32)
        logits_neg = torch.tensor([1.0, 0.0], dtype=torch.float32)
        bce_all = torch.cat(
            [
                F.binary_cross_entropy_with_logits(logits_pos, torch.ones_like(logits_pos), reduction="none"),
                F.binary_cross_entropy_with_logits(logits_neg, torch.zeros_like(logits_neg), reduction="none"),
            ],
            dim=0,
        )
        row_w_pair = torch.tensor([2.0, 3.0, 6.0, 6.0], dtype=torch.float32)
        row_w_pair = row_w_pair / row_w_pair.mean()
        expected = (bce_all * row_w_pair).sum() / row_w_pair.sum()

        self.assertAlmostEqual(stats["bce_weighted"], float(expected.item()), places=6)


class TestTrainSbiNreSplits(unittest.TestCase):
    def test_cluster_holdout_excludes_whole_clusters_from_trainval(self):
        cluster_ids = np.array([-1, -1, 1, 1, 2, 2, 3, 3, 4, 4], dtype=np.int64)
        trainval_idx, test_idx, heldout = _compute_test_split_with_cluster_holdout(
            n_total=len(cluster_ids),
            test_split=0.2,
            cluster_ids=cluster_ids,
            test_cluster_frac=0.5,
            random_state=7,
        )

        self.assertGreaterEqual(len(heldout), 1)
        heldout_mask = np.isin(cluster_ids, heldout)
        self.assertFalse(np.any(heldout_mask[trainval_idx]))
        self.assertTrue(np.all(np.isin(np.where(heldout_mask)[0], test_idx)))

    def test_zero_test_split_returns_all_rows_when_no_cluster_holdout(self):
        trainval_idx, test_idx, heldout = _compute_test_split_with_cluster_holdout(
            n_total=8,
            test_split=0.0,
            cluster_ids=None,
            test_cluster_frac=0.0,
            random_state=11,
        )

        np.testing.assert_array_equal(trainval_idx, np.arange(8, dtype=np.int64))
        self.assertEqual(len(test_idx), 0)
        self.assertEqual(len(heldout), 0)


if __name__ == "__main__":
    unittest.main()
