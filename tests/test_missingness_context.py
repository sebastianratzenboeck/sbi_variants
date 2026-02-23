import unittest

import torch

from transformer import Simformer


class TestMissingnessContextToken(unittest.TestCase):
    def _make_model(self, use_missingness_context: bool):
        return Simformer(
            num_nodes=10,
            dim_value=8,
            dim_id=8,
            dim_condition=4,
            dim_error=4,
            dim_observed=4,
            attn_embed_dim=32,
            num_heads=4,
            num_layers=2,
            widening_factor=2,
            time_embed_dim=16,
            dropout=0.0,
            use_missingness_context=use_missingness_context,
            obs_start_idx=4,
            survey_obs_groups=[[4, 5], [6, 7, 8, 9]],
            missingness_context_hidden_dim=16,
        )

    def _inputs(self, batch_size=3, num_nodes=10):
        t = torch.rand(batch_size, 1, 1)
        x = torch.randn(batch_size, num_nodes, 1)
        node_ids = torch.arange(num_nodes).unsqueeze(0).expand(batch_size, -1)
        condition_mask = torch.zeros(batch_size, num_nodes, 1)
        edge_mask = torch.ones(batch_size, num_nodes, num_nodes, dtype=torch.bool)

        observed_mask = torch.ones(batch_size, num_nodes)
        observed_mask[:, 7:] = 0.0
        errors = torch.zeros(batch_size, num_nodes)
        errors[:, 4:7] = -0.5   # real-error regime
        errors[:, 7:] = 5.0     # unobserved sentinel
        return t, x, node_ids, condition_mask, edge_mask, errors, observed_mask

    def test_forward_shape_with_missingness_token(self):
        model = self._make_model(use_missingness_context=True)
        args = self._inputs()
        out = model(*args)
        self.assertEqual(tuple(out.shape), (3, 10, 1))

    def test_forward_shape_without_missingness_token(self):
        model = self._make_model(use_missingness_context=False)
        args = self._inputs()
        out = model(*args)
        self.assertEqual(tuple(out.shape), (3, 10, 1))


if __name__ == "__main__":
    unittest.main()
