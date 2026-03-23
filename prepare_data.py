#!/usr/bin/env python
"""
Prepare mock galaxy data for SimFormer training.

Takes raw mock galaxy catalog (CSV or Parquet), applies quality cuts,
generates synthetic errors and observed values for non-Gaia surveys,
and outputs a Parquet file with the same column structure as real
survey data.

When real data with actual _obs/_err columns becomes available,
this script is no longer needed — the training script reads the
real Parquet directly.

Usage:
  python prepare_data.py --data-path /path/to/train_data.csv --output-path prepared.parquet
  python prepare_data.py --data-path /path/to/train_data.parquet --output-path prepared.parquet --max-rad 5.0
"""

import argparse
import numpy as np
import pandas as pd

try:
    from .columns import SURVEY_ERRORS
except ImportError:
    from columns import SURVEY_ERRORS


def galactic_to_unitvec(glon_deg, glat_deg):
    """Convert Galactic coordinates (l, b) in degrees to 3D unit vectors.

    Avoids periodicity issues at l=0/360 and pole singularities at b=±90
    by representing sky position as a Cartesian unit vector (ux, uy, uz).

    Args:
        glon_deg: Array of Galactic longitude in degrees [0, 360).
        glat_deg: Array of Galactic latitude in degrees [-90, 90].

    Returns:
        ux, uy, uz: Arrays of unit-vector components, each same shape as input.
    """
    l_rad = np.deg2rad(glon_deg)
    b_rad = np.deg2rad(glat_deg)
    cos_b = np.cos(b_rad)
    ux = cos_b * np.cos(l_rad)
    uy = cos_b * np.sin(l_rad)
    uz = np.sin(b_rad)
    return ux, uy, uz


def load_and_filter(data_path, max_rad, min_log_age=5.0):
    """Load CSV or Parquet file, apply distance and age cuts."""
    print(f'Loading data from {data_path} ...')
    if data_path.endswith('.parquet'):
        df = pd.read_parquet(data_path)
    else:
        df = pd.read_csv(data_path)
    print(f'  Loaded {len(df):,} stars')
    df = df[(df.rad < max_rad) & (df.logAge > min_log_age)].reset_index(drop=True)
    print(f'  After cuts (rad < {max_rad}, logAge > {min_log_age}): {len(df):,} stars')
    return df


def generate_synthetic_errors(df, unobs_frac, seed=42):
    """Generate synthetic _obs and _err columns for non-Gaia surveys.

    For each survey, adds Gaussian noise to true magnitudes to create
    observed values, and randomly masks a fraction of stars as unobserved
    (NaN). All bands within a survey share the same missingness pattern.
    """
    rng = np.random.RandomState(seed)
    N = len(df)

    for survey_name, (bands, sigma) in SURVEY_ERRORS.items():
        # Per-survey unobserved mask (same for all bands in the survey)
        is_unobserved = rng.rand(N) < unobs_frac

        for band in bands:
            true_val = df[band].values.copy()
            noise = rng.randn(N) * sigma
            obs_val = true_val + noise
            err_val = np.full(N, sigma)

            # Set unobserved stars to NaN
            obs_val[is_unobserved] = np.nan
            err_val[is_unobserved] = np.nan

            obs_col = band.replace('_mag', '_mag_obs')
            err_col = band.replace('_mag', '_mag_err')
            df[obs_col] = obs_val
            df[err_col] = err_val

    print(f'  Synthetic errors created (unobs_frac={unobs_frac})')
    for survey_name, (bands, _) in SURVEY_ERRORS.items():
        obs_col = bands[0].replace('_mag', '_mag_obs')
        n_nan = df[obs_col].isna().sum()
        print(f'    {survey_name}: {n_nan:,} / {N:,} unobserved ({100*n_nan/N:.1f}%)')
    return df


def parse_args():
    p = argparse.ArgumentParser(
        description='Prepare mock galaxy data for SimFormer training.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument('--data-path', type=str, required=True,
                   help='Path to raw mock galaxy catalog (CSV or Parquet)')
    p.add_argument('--output-path', type=str, required=True,
                   help='Output path for prepared Parquet file')
    p.add_argument('--max-rad', type=float, default=5.0,
                   help='Distance cut in kpc')
    p.add_argument('--min-log-age', type=float, default=5.0,
                   help='Minimum logAge cut')
    p.add_argument('--unobs-frac', type=float, default=0.20,
                   help='Fraction of stars unobserved per non-Gaia survey')
    p.add_argument('--seed', type=int, default=42,
                   help='Random seed for reproducibility')
    return p.parse_args()


def main():
    args = parse_args()
    np.random.seed(args.seed)

    print('\n--- Data Preparation ---')
    df = load_and_filter(args.data_path, args.max_rad, args.min_log_age)
    df = generate_synthetic_errors(df, args.unobs_frac, seed=args.seed)

    # Save as Parquet
    df.to_parquet(args.output_path, index=False)
    print(f'\n  Saved {len(df):,} stars to {args.output_path}')
    print(f'  Columns: {len(df.columns)}')


if __name__ == '__main__':
    main()
