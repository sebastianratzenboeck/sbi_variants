import json
import os
import tempfile
import types
import unittest

import numpy as np
import torch

from columns import ALL_VALUE_COLS, N_INTRINSIC, N_TRUE_MAG, NUM_NODES
from columns import OBS_COLS, TRUE_MAG_COLS
from inference_utils import NormStats
from sample_mock_galaxy import prepare_observations_from_cache, load_model, sample_posterior
from encoder import ObservationEncoder
from posterior_models import ConditionalFMPosterior
from transformer import Simformer


class TestSampleMockGalaxyCache(unittest.TestCase):
    def _make_cache(self, path, values, errors, observed, columns=None):
        np.savez(
            path,
            values_norm=values,
            errors_norm=errors,
            observed_mask=observed,
            columns=np.array(columns if columns is not None else ALL_VALUE_COLS),
        )

    def test_prepare_observations_from_cache_condition_mask_semantics(self):
        obs_start = N_INTRINSIC + N_TRUE_MAG

        values = np.arange(3 * NUM_NODES, dtype=np.float32).reshape(3, NUM_NODES)
        errors = np.ones((3, NUM_NODES), dtype=np.float32)

        observed = np.ones((3, NUM_NODES), dtype=np.float32)
        observed[:, obs_start:] = 0.0
        # Configure row 2 observed-data pattern explicitly
        observed[2, obs_start + 0] = 1.0
        observed[2, obs_start + 1] = 0.0
        observed[2, obs_start + 2] = 1.0

        with tempfile.TemporaryDirectory() as td:
            cache_path = os.path.join(td, "cache.npz")
            self._make_cache(cache_path, values, errors, observed)

            cv, cm, om, er, selected = prepare_observations_from_cache(
                cache_path=cache_path,
                indices=np.array([2, 0], dtype=np.int64),
                max_stars=1,
                device="cpu",
            )

        self.assertEqual(tuple(cv.shape), (1, NUM_NODES))
        self.assertEqual(tuple(cm.shape), (1, NUM_NODES, 1))
        self.assertEqual(tuple(om.shape), (1, NUM_NODES))
        self.assertEqual(tuple(er.shape), (1, NUM_NODES))
        self.assertListEqual(selected.tolist(), [2])

        # Sky is always conditioned
        self.assertTrue(torch.all(cm[0, 0:3, 0] == 1.0).item())
        # Intrinsic non-sky remains unconditioned
        self.assertEqual(cm[0, 3, 0].item(), 0.0)
        # Obs block condition mask mirrors observed mask
        self.assertEqual(cm[0, obs_start + 0, 0].item(), 1.0)
        self.assertEqual(cm[0, obs_start + 1, 0].item(), 0.0)
        self.assertEqual(cm[0, obs_start + 2, 0].item(), 1.0)

        # Row selection should preserve values/errors from selected row
        torch.testing.assert_close(cv[0], torch.tensor(values[2]))
        torch.testing.assert_close(er[0], torch.tensor(errors[2]))

    def test_prepare_observations_from_cache_rejects_mismatched_columns(self):
        values = np.zeros((1, NUM_NODES), dtype=np.float32)
        errors = np.zeros((1, NUM_NODES), dtype=np.float32)
        observed = np.ones((1, NUM_NODES), dtype=np.float32)

        bad_columns = list(ALL_VALUE_COLS)
        bad_columns[0] = "not_a_real_column"

        with tempfile.TemporaryDirectory() as td:
            cache_path = os.path.join(td, "bad_columns.npz")
            self._make_cache(cache_path, values, errors, observed, columns=bad_columns)

            with self.assertRaises(ValueError):
                prepare_observations_from_cache(cache_path=cache_path, device="cpu")

    def test_prepare_observations_from_cache_rejects_out_of_bounds_indices(self):
        values = np.zeros((2, NUM_NODES), dtype=np.float32)
        errors = np.zeros((2, NUM_NODES), dtype=np.float32)
        observed = np.ones((2, NUM_NODES), dtype=np.float32)

        with tempfile.TemporaryDirectory() as td:
            cache_path = os.path.join(td, "cache.npz")
            self._make_cache(cache_path, values, errors, observed)

            with self.assertRaises(ValueError):
                prepare_observations_from_cache(
                    cache_path=cache_path,
                    indices=np.array([10], dtype=np.int64),
                    device="cpu",
                )

    def test_prepare_observations_from_cache_can_select_expected_column_subset(self):
        values = np.arange(2 * NUM_NODES, dtype=np.float32).reshape(2, NUM_NODES)
        errors = np.ones((2, NUM_NODES), dtype=np.float32)
        observed = np.ones((2, NUM_NODES), dtype=np.float32)
        obs_start = N_INTRINSIC + N_TRUE_MAG
        observed[:, obs_start:] = 0.0
        observed[:, obs_start + 0] = 1.0

        expected_columns = [c for c in ALL_VALUE_COLS if c not in TRUE_MAG_COLS]
        expected_idx = np.array([ALL_VALUE_COLS.index(c) for c in expected_columns], dtype=np.int64)

        with tempfile.TemporaryDirectory() as td:
            cache_path = os.path.join(td, "cache_subset.npz")
            self._make_cache(cache_path, values, errors, observed)

            cv, cm, om, er, selected = prepare_observations_from_cache(
                cache_path=cache_path,
                expected_columns=expected_columns,
                device="cpu",
            )

        self.assertEqual(tuple(cv.shape), (2, len(expected_columns)))
        self.assertEqual(tuple(cm.shape), (2, len(expected_columns), 1))
        self.assertEqual(tuple(om.shape), (2, len(expected_columns)))
        self.assertEqual(tuple(er.shape), (2, len(expected_columns)))
        self.assertListEqual(selected.tolist(), [0, 1])
        torch.testing.assert_close(cv, torch.tensor(values[:, expected_idx]))
        torch.testing.assert_close(er, torch.tensor(errors[:, expected_idx]))

        # Sky always conditioned in reduced layout.
        for sky in ("sky_ux", "sky_uy", "sky_uz"):
            i = expected_columns.index(sky)
            self.assertTrue(torch.all(cm[:, i, 0] == 1.0).item())

        # Observation columns mirror observed mask in condition mask.
        for col in OBS_COLS:
            if col in expected_columns:
                i = expected_columns.index(col)
                torch.testing.assert_close(cm[:, i, 0], om[:, i])


class TestLoadModelCheckpointCompat(unittest.TestCase):
    def _write_norm_stats(self, path, n):
        np.savez(
            path,
            means=np.zeros(n, dtype=np.float32),
            stds=np.ones(n, dtype=np.float32),
            columns=np.array([f"c{i}" for i in range(n)]),
            log_err_mean=np.array(0.0, dtype=np.float32),
            log_err_std=np.array(1.0, dtype=np.float32),
        )

    def _make_config(self):
        return {
            "num_nodes": 8,
            "dim_value": 4,
            "dim_id": 4,
            "dim_condition": 2,
            "dim_error": 2,
            "dim_observed": 2,
            "attn_embed_dim": 16,
            "num_heads": 2,
            "num_layers": 1,
            "widening_factor": 2,
            "time_embed_dim": 8,
            "dropout": 0.0,
        }

    def _make_posterior_config(self):
        return {
            "method": "flow_matching",
            "input_columns": [f"c{i}" for i in range(8)],
            "theta_columns": ["c0", "c1", "c2"],
            "dim_value": 4,
            "dim_id": 4,
            "dim_error": 2,
            "dim_observed": 2,
            "attn_embed_dim": 16,
            "num_heads": 2,
            "num_layers": 1,
            "widening_factor": 2,
            "dropout": 0.0,
            "fm_hidden_dim": 16,
            "time_embed_dim": 8,
            "sigma_min": 1e-3,
            "time_prior_exponent": 0.0,
        }

    def test_load_model_accepts_prefixed_compiled_state_dict(self):
        run_name = "testrun"
        config = self._make_config()

        model = Simformer(**config)
        base_state = model.state_dict()
        prefixed_state = {f"_orig_mod.{k}": v.clone() for k, v in base_state.items()}

        with tempfile.TemporaryDirectory() as td:
            with open(os.path.join(td, f"model_config_{run_name}.json"), "w") as f:
                json.dump(config, f)
            torch.save(prefixed_state, os.path.join(td, f"best_model_{run_name}.pt"))
            self._write_norm_stats(os.path.join(td, "norm_stats.npz"), config["num_nodes"])

            loaded_model, stats = load_model(td, run_name=run_name, device="cpu")

        self.assertEqual(stats.num_nodes, config["num_nodes"])
        loaded_state = loaded_model.state_dict()
        for k, v in base_state.items():
            torch.testing.assert_close(loaded_state[k], v)

    def test_load_model_supports_posterior_artifact_layout(self):
        run_name = "testrun"
        config = self._make_posterior_config()

        encoder = ObservationEncoder(
            input_columns=config["input_columns"],
            dim_value=config["dim_value"],
            dim_id=config["dim_id"],
            value_calibration_type="scalar_film",
            dim_error=config["dim_error"],
            error_embed_type="mlp_regime",
            dim_observed=config["dim_observed"],
            attn_embed_dim=config["attn_embed_dim"],
            num_heads=config["num_heads"],
            num_layers=config["num_layers"],
            widening_factor=config["widening_factor"],
            dropout=config["dropout"],
            use_missingness_context=False,
            missingness_context_hidden_dim=64,
        )
        model = ConditionalFMPosterior(
            encoder=encoder,
            theta_dim=len(config["theta_columns"]),
            hidden_dim=config["fm_hidden_dim"],
            time_embed_dim=config["time_embed_dim"],
            sigma_min=config["sigma_min"],
            time_prior_exponent=config["time_prior_exponent"],
            dropout=config["dropout"],
        )
        base_state = model.state_dict()
        prefixed_state = {f"_orig_mod.{k}": v.clone() for k, v in base_state.items()}

        with tempfile.TemporaryDirectory() as td:
            with open(os.path.join(td, f"posterior_config_{run_name}.json"), "w") as f:
                json.dump(config, f)
            torch.save(prefixed_state, os.path.join(td, f"best_model_{run_name}.pt"))
            self._write_norm_stats(
                os.path.join(td, f"posterior_norm_meta_{run_name}.npz"),
                len(config["input_columns"]),
            )

            loaded_model, stats = load_model(td, run_name=run_name, device="cpu")

        self.assertEqual(stats.num_nodes, len(config["input_columns"]))
        loaded_state = loaded_model.state_dict()
        for k, v in base_state.items():
            torch.testing.assert_close(loaded_state[k], v)


class TestDirectPosteriorSamplingCompat(unittest.TestCase):
    def _make_model(self, input_columns, theta_dim):
        encoder = ObservationEncoder(
            input_columns=input_columns,
            dim_value=4,
            dim_id=4,
            value_calibration_type="scalar_film",
            dim_error=2,
            error_embed_type="mlp_regime",
            dim_observed=2,
            attn_embed_dim=16,
            num_heads=2,
            num_layers=1,
            widening_factor=2,
            dropout=0.0,
            use_missingness_context=False,
            missingness_context_hidden_dim=64,
        )
        return ConditionalFMPosterior(
            encoder=encoder,
            theta_dim=theta_dim,
            hidden_dim=16,
            time_embed_dim=8,
            sigma_min=1e-3,
            time_prior_exponent=0.0,
            dropout=0.0,
        )

    def test_sample_posterior_uses_direct_posterior_input_subset_and_embeds_theta(self):
        model = self._make_model(input_columns=["c3", "c4"], theta_dim=3)
        model._sbi_artifact_kind = "direct_posterior"
        model._sbi_input_columns = ["c3", "c4"]
        model._sbi_theta_columns = ["c0", "c1", "c2"]
        model._sbi_theta_indices = [0, 1, 2]
        model._sbi_full_columns = [f"c{i}" for i in range(8)]
        model._sbi_full_dim = 8
        model._sbi_norm_stats = object()

        captured = {}

        def fake_sample(self, values, errors, observed_mask, num_samples, steps):
            captured["values"] = values.detach().cpu()
            captured["errors"] = errors.detach().cpu()
            captured["observed"] = observed_mask.detach().cpu()
            return torch.full((values.shape[0], num_samples, 3), 7.0, device=values.device)

        model.sample = types.MethodType(fake_sample, model)

        condition_values = torch.arange(16, dtype=torch.float32).reshape(2, 8)
        errors = torch.full((2, 8), 0.5, dtype=torch.float32)
        observed = torch.ones((2, 8), dtype=torch.float32)
        condition_mask = torch.ones((2, 8, 1), dtype=torch.float32)

        samples = sample_posterior(
            model=model,
            condition_values=condition_values,
            condition_mask=condition_mask,
            observed_mask=observed,
            errors=errors,
            num_samples=4,
            batch_size=2,
            steps=12,
            device="cpu",
        )

        self.assertEqual(tuple(captured["values"].shape), (2, 2))
        torch.testing.assert_close(captured["values"], condition_values[:, [3, 4]])
        torch.testing.assert_close(captured["errors"], errors[:, [3, 4]])
        torch.testing.assert_close(captured["observed"], observed[:, [3, 4]])

        self.assertEqual(tuple(samples.shape), (2, 4, 8))
        torch.testing.assert_close(samples[:, :, 0:3], torch.full((2, 4, 3), 7.0))
        repeated = condition_values.unsqueeze(1).repeat(1, 4, 1)
        torch.testing.assert_close(samples[:, :, 3:], repeated[:, :, 3:])

    def test_sample_posterior_reconstructs_color_inputs_for_direct_posterior(self):
        model = self._make_model(
            input_columns=["GAIA_GAIA3.Gbp_mag_obs", "color_BP_RP"],
            theta_dim=3,
        )
        model._sbi_artifact_kind = "direct_posterior"
        model._sbi_input_columns = ["GAIA_GAIA3.Gbp_mag_obs", "color_BP_RP"]
        model._sbi_theta_columns = ["feh", "m_init", "logAge"]
        model._sbi_theta_indices = [3, 4, 5]
        model._sbi_full_columns = list(ALL_VALUE_COLS)
        model._sbi_full_dim = len(ALL_VALUE_COLS)

        with tempfile.TemporaryDirectory() as td:
            norm_path = os.path.join(td, "posterior_norm_meta_test.npz")
            transform_names = np.asarray(["identity"] * len(ALL_VALUE_COLS), dtype=object)
            transform_names[ALL_VALUE_COLS.index("rad")] = "log_shifted_pos"
            transform_names[ALL_VALUE_COLS.index("Av")] = "log1p_pos"
            np.savez(
                norm_path,
                means=np.zeros(len(ALL_VALUE_COLS), dtype=np.float32),
                stds=np.ones(len(ALL_VALUE_COLS), dtype=np.float32),
                columns=np.asarray(ALL_VALUE_COLS, dtype=object),
                value_transform_names=transform_names,
                value_transform_params=np.zeros(len(ALL_VALUE_COLS), dtype=np.float32),
                log_err_mean=np.array(0.0, dtype=np.float32),
                log_err_std=np.array(1.0, dtype=np.float32),
                input_columns=np.asarray(["GAIA_GAIA3.Gbp_mag_obs", "color_BP_RP"], dtype=object),
                use_colors=np.array(True, dtype=bool),
                color_names=np.asarray(["color_BP_RP"], dtype=object),
                color_means=np.asarray([0.0], dtype=np.float32),
                color_stds=np.asarray([1.0], dtype=np.float32),
            )
            model._sbi_norm_stats = NormStats(norm_path)

        captured = {}

        def fake_sample(self, values, errors, observed_mask, num_samples, steps):
            captured["values"] = values.detach().cpu()
            captured["errors"] = errors.detach().cpu()
            captured["observed"] = observed_mask.detach().cpu()
            return torch.zeros((values.shape[0], num_samples, 3), device=values.device)

        model.sample = types.MethodType(fake_sample, model)

        condition_values = torch.zeros((1, len(ALL_VALUE_COLS)), dtype=torch.float32)
        errors = torch.full((1, len(ALL_VALUE_COLS)), 0.0, dtype=torch.float32)
        observed = torch.zeros((1, len(ALL_VALUE_COLS)), dtype=torch.float32)
        condition_mask = torch.zeros((1, len(ALL_VALUE_COLS), 1), dtype=torch.float32)

        idx_bp = ALL_VALUE_COLS.index("GAIA_GAIA3.Gbp_mag_obs")
        idx_rp = ALL_VALUE_COLS.index("GAIA_GAIA3.Grp_mag_obs")
        condition_values[0, idx_bp] = 2.0
        condition_values[0, idx_rp] = 0.5
        observed[0, idx_bp] = 1.0
        observed[0, idx_rp] = 1.0

        sample_posterior(
            model=model,
            condition_values=condition_values,
            condition_mask=condition_mask,
            observed_mask=observed,
            errors=errors,
            num_samples=2,
            batch_size=1,
            steps=4,
            device="cpu",
        )

        self.assertEqual(tuple(captured["values"].shape), (1, 2))
        self.assertAlmostEqual(float(captured["values"][0, 0]), 2.0, places=6)
        self.assertAlmostEqual(float(captured["values"][0, 1]), 1.5, places=6)
        self.assertAlmostEqual(float(captured["errors"][0, 1]), float(np.log(np.sqrt(2.0))), places=6)
        self.assertEqual(float(captured["observed"][0, 1]), 1.0)


if __name__ == "__main__":
    unittest.main()
