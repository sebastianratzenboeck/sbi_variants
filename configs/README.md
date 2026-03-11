# Config Notes

Most JSON files are consumed by `train_sbi_posterior.py`.
`train_nre_balanced_theta.json` is consumed by `train_sbi_nre.py`.

`train_nre_balanced_theta.json` also includes optional joint curriculum keys:
- `joint_curriculum`
- `n_bins`, `n_mass_bins`, `curriculum_bin_strategy`
- `tau_max`, `tau_warmup`, `curriculum_epoch_size`
- `curriculum_importance_weighting` (+ beta/min/max)
They must stay strict JSON (no inline comments), so notes are documented here.

## Curriculum / Binning

- `train_sbi_posterior.py` defaults to `--curriculum-bin-strategy quantile`.
- Existing config files may still pin `tau_max` to `0.0` for fixed-uniform curriculum.
  In that case, the trainer runs fixed-uniform mode and `tau_warmup` has no effect.

## Flow-Matching Legacy Clip Bounds

- Some FM-oriented configs use legacy importance clip bounds:
  - `importance_weight_min = 0.5`
  - `importance_weight_max = 2.0`
- Runtime compatibility behavior in `train_sbi_posterior.py`:
  for `--method flow_matching` with `importance_weighting=true` and
  `importance_weight_beta=0.25`, these legacy bounds are auto-widened at runtime to:
  - `importance_weight_min = 0.1`
  - `importance_weight_max = 10.0`
- The trainer prints a startup message when this auto-widening is applied.
