#!/usr/bin/env python
"""Create a test-only cache from build_arrays_cache.npz and test_indices.npy.

This script extracts only rows listed in an index file (typically test indices)
and writes a smaller cache file that can be loaded quickly for evaluation.
"""

from __future__ import annotations

import argparse
import os
import time

import numpy as np


REQUIRED_ARRAY_KEYS = ("values_norm", "errors_norm", "observed_mask")
OPTIONAL_META_KEYS = (
    "means",
    "stds",
    "log_err_mean",
    "log_err_std",
    "columns",
    "value_transform_names",
    "value_transform_params",
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Build a test-only cache from full cache + index file.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--cache-path", type=str, required=True,
                   help="Path to full build_arrays_cache.npz")
    p.add_argument("--index-file", type=str, required=True,
                   help="Path to index file (e.g., test_indices.npy)")
    p.add_argument("--output-path", type=str, default=None,
                   help="Output .npz path (default: <cache-dir>/build_arrays_cache_test.npz)")
    p.add_argument("--compress", action="store_true", default=False,
                   help="Write compressed NPZ (smaller, slower)")
    p.add_argument("--sort-indices", action="store_true", default=False,
                   help="Sort index array before extraction (optional)")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if not os.path.isfile(args.cache_path):
        raise FileNotFoundError(f"cache file not found: {args.cache_path}")
    if not os.path.isfile(args.index_file):
        raise FileNotFoundError(f"index file not found: {args.index_file}")

    output_path = args.output_path
    if output_path is None:
        cache_dir = os.path.dirname(os.path.abspath(args.cache_path))
        output_path = os.path.join(cache_dir, "build_arrays_cache_test.npz")

    print("--- Loading indices ---")
    idx = np.load(args.index_file).astype(np.int64)
    if idx.ndim != 1:
        raise ValueError(f"Expected 1-D indices, got shape {idx.shape}")
    if idx.size == 0:
        raise ValueError("Index file is empty.")
    if args.sort_indices:
        idx = np.sort(idx)
    print(f"  indices: {idx.size:,}")
    print(f"  min/max: {idx.min()} / {idx.max()}")

    t0 = time.time()
    print("\n--- Opening full cache ---")
    d = np.load(args.cache_path, allow_pickle=True)
    missing = [k for k in REQUIRED_ARRAY_KEYS if k not in d]
    if missing:
        raise KeyError(f"Missing required keys in cache: {missing}")

    n_total = d["values_norm"].shape[0]
    if idx.min() < 0 or idx.max() >= n_total:
        raise ValueError(
            f"indices out of bounds for cache rows={n_total:,}: "
            f"min={idx.min()}, max={idx.max()}"
        )
    print(f"  cache rows: {n_total:,}")

    out = {}
    for key in REQUIRED_ARRAY_KEYS:
        print(f"\n--- Extracting {key} ---")
        arr = d[key]  # loads full array from NPZ
        print(f"  full shape: {arr.shape}, dtype={arr.dtype}")
        out[key] = arr[idx]
        print(f"  test shape: {out[key].shape}, dtype={out[key].dtype}")
        del arr

    for key in OPTIONAL_META_KEYS:
        if key in d:
            out[key] = d[key]

    # Keep provenance
    out["selected_indices"] = idx
    out["source_cache_path"] = np.array(str(args.cache_path))
    out["source_index_file"] = np.array(str(args.index_file))

    saver = np.savez_compressed if args.compress else np.savez
    print("\n--- Writing output ---")
    print(f"  output: {output_path}")
    print(f"  compressed: {args.compress}")
    saver(output_path, **out)

    # Light verification
    print("\n--- Verifying output ---")
    check = np.load(output_path, allow_pickle=True)
    for key in REQUIRED_ARRAY_KEYS:
        print(f"  {key}: {check[key].shape}, dtype={check[key].dtype}")
    print(f"  selected_indices: {check['selected_indices'].shape}")

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.1f}s")


if __name__ == "__main__":
    main()
