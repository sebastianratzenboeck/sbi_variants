#!/usr/bin/env python
"""Create paper-oriented plots from evaluation CSV artifacts."""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import tempfile
from typing import Iterable

import numpy as np
import pandas as pd

# Avoid matplotlib cache warnings on systems where ~/.matplotlib is not writable.
os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "mplconfig"))
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(
        description="Plot coverage/TARP/sensitivity evaluation outputs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--eval-dir", type=str, required=True,
                   help="Directory containing eval_*.csv outputs")
    p.add_argument("--output-dir", type=str, default=None,
                   help="Output plot directory (default: <eval-dir>/plots)")
    p.add_argument("--fmt", type=str, default="png",
                   help="Output image format: png or pdf")
    p.add_argument("--dpi", type=int, default=180)

    p.add_argument("--coverage-file", type=str, default=None)
    p.add_argument("--tarp-file", type=str, default=None)
    p.add_argument("--sensitivity-main-file", type=str, default=None)
    p.add_argument("--sensitivity-bands-file", type=str, default=None)
    p.add_argument("--sensitivity-snr-file", type=str, default=None)
    p.add_argument("--per-param-bands-file", type=str, default=None)
    p.add_argument("--per-param-snr-file", type=str, default=None)
    p.add_argument("--params", type=str, default="logAge,feh,logg,logT,rad",
                   help="Comma-separated parameters for per-parameter plots")
    return p.parse_args()


def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def first_existing(paths: Iterable[str]) -> str | None:
    for p in paths:
        if p and os.path.isfile(p):
            return p
    return None


def latest_by_glob(eval_dir: str, patterns: list[str]) -> str | None:
    matches = []
    for pat in patterns:
        matches.extend(glob.glob(os.path.join(eval_dir, pat)))
    if not matches:
        return None
    matches.sort(key=os.path.getmtime, reverse=True)
    return matches[0]


def resolve_file(explicit: str | None, eval_dir: str, patterns: list[str]) -> str | None:
    if explicit is not None:
        if not os.path.isfile(explicit):
            raise FileNotFoundError(f"File not found: {explicit}")
        return explicit
    return latest_by_glob(eval_dir, patterns)


def paired_per_param_path(summary_path: str | None) -> str | None:
    if not summary_path:
        return None
    base = os.path.basename(summary_path)
    if not base.startswith("sensitivity_summary_"):
        return None
    suffix = base[len("sensitivity_summary_"):]
    candidate = os.path.join(os.path.dirname(summary_path), f"sensitivity_per_param_{suffix}")
    return candidate if os.path.isfile(candidate) else None


def infer_macro_cols(df: pd.DataFrame) -> tuple[str, str]:
    cov_cols = [c for c in df.columns if c.startswith("coverage_") and c.endswith("_macro")]
    width_cols = [c for c in df.columns if c.startswith("width_") and c.endswith("_macro")]
    if not cov_cols or not width_cols:
        raise ValueError("Could not infer coverage/width macro columns from sensitivity summary.")
    return cov_cols[0], width_cols[0]


def infer_param_cols(df: pd.DataFrame) -> tuple[str, str]:
    cov_cols = [c for c in df.columns if c.startswith("coverage_")]
    width_cols = [c for c in df.columns if c.startswith("width_")]
    if not cov_cols or not width_cols:
        raise ValueError("Could not infer coverage/width columns from per-param sensitivity.")
    return cov_cols[0], width_cols[0]


def scenario_family(name: str) -> str:
    if name == "baseline":
        return "baseline"
    if name.startswith("maskdrop_"):
        return "maskdrop"
    if name.startswith("errscale_"):
        return "errscale"
    if name.startswith("survey_"):
        return "survey"
    return "other"


def scenario_label(name: str) -> str:
    if name == "baseline":
        return "full (baseline)"
    if name.startswith("survey_"):
        mode = name.replace("survey_", "")
        return mode.replace("_", "+")
    return name


def savefig(fig, out_path: str, dpi: int):
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_coverage(df: pd.DataFrame, out_path: str, dpi: int):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

    ax = axes[0]
    ax.plot(df["level"], df["mean_coverage"], marker="o", lw=2, label="Empirical")
    ax.plot([0, 1], [0, 1], ls="--", c="black", lw=1.5, label="Ideal")
    ax.set_xlabel("Nominal central coverage")
    ax.set_ylabel("Empirical coverage")
    ax.set_title("1D Coverage Calibration")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)

    ax = axes[1]
    y = df["mean_abs_calibration_error"]
    ax.bar(df["level"].astype(str), y, color="#3A78B2")
    ax.set_xlabel("Nominal central coverage")
    ax.set_ylabel("Abs calibration error")
    ax.set_title(f"ACE by Level (mean={y.mean():.3f})")
    ax.grid(axis="y", alpha=0.25)

    savefig(fig, out_path, dpi)


def plot_tarp(df: pd.DataFrame, out_path: str, dpi: int):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

    ax = axes[0]
    yerr = None
    if "per_projection_std" in df.columns:
        yerr = df["per_projection_std"].values
    ax.errorbar(
        df["alpha"],
        df["empirical_coverage"],
        yerr=yerr,
        marker="o",
        lw=2,
        capsize=3,
        color="#2A9D8F",
        label="TARP empirical",
    )
    ax.plot([0, 1], [0, 1], ls="--", c="black", lw=1.5, label="Ideal")
    ax.set_xlabel("Nominal central coverage (alpha)")
    ax.set_ylabel("Empirical coverage")
    ax.set_title("TARP Calibration Curve")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)

    ax = axes[1]
    cal_err = df["calibration_error"].values
    ax.plot(df["alpha"], cal_err, marker="o", lw=2, color="#E76F51")
    ax.axhline(0.0, ls="--", c="black", lw=1.0)
    ax.set_xlabel("alpha")
    ax.set_ylabel("Empirical - nominal")
    ax.set_title(f"TARP Calibration Error (ACE={np.mean(np.abs(cal_err)):.3f})")
    ax.grid(alpha=0.25)

    savefig(fig, out_path, dpi)


def plot_frontier(df: pd.DataFrame, out_path: str, dpi: int):
    cov_col, width_col = infer_macro_cols(df)
    colors = {
        "baseline": "#1b9e77",
        "errscale": "#d95f02",
        "survey": "#7570b3",
        "maskdrop": "#66a61e",
        "other": "#999999",
    }
    fam = df["scenario"].map(scenario_family)

    fig, ax = plt.subplots(figsize=(7.5, 5.2))
    for f in sorted(fam.unique()):
        m = fam == f
        ax.scatter(
            df.loc[m, width_col],
            df.loc[m, "rmse_norm_macro"],
            label=f,
            s=70,
            alpha=0.85,
            c=colors.get(f, "#999999"),
            edgecolors="black",
            linewidths=0.3,
        )

    for _, r in df.iterrows():
        ax.annotate(r["scenario"], (r[width_col], r["rmse_norm_macro"]), fontsize=7, alpha=0.9)

    ax.set_xlabel(f"Posterior width ({width_col})")
    ax.set_ylabel("RMSE (normalized)")
    ax.set_title(f"Sharpness vs Accuracy (color by scenario family, coverage={cov_col})")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=8)
    savefig(fig, out_path, dpi)


def band_order(name: str) -> int:
    # Lower to higher information.
    table = {
        "survey_gaia_only": 0,
        "survey_gaia_2mass": 1,
        "survey_full": 2,
        "baseline": 2,
    }
    return table.get(name, 999)


def plot_band_ladder(df: pd.DataFrame, out_path: str, dpi: int):
    cov_col, width_col = infer_macro_cols(df)
    m = df["scenario"].eq("baseline") | df["scenario"].str.startswith("survey_")
    b = df[m].copy()
    if b.empty:
        return
    b["order"] = b["scenario"].map(band_order)
    b = b.sort_values("order")
    b["label"] = b["scenario"].map(scenario_label)
    x = np.arange(len(b))

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.4))

    axes[0].plot(x, b["rmse_norm_macro"], marker="o", lw=2, color="#264653")
    axes[0].set_ylabel("RMSE (normalized)")
    axes[0].set_title("Accuracy")
    axes[0].grid(alpha=0.25)

    axes[1].plot(x, b[width_col], marker="o", lw=2, color="#2A9D8F")
    axes[1].set_ylabel(width_col)
    axes[1].set_title("Posterior Width")
    axes[1].grid(alpha=0.25)

    axes[2].plot(x, b[cov_col], marker="o", lw=2, color="#E76F51")
    axes[2].axhline(float(re.findall(r"\d+", cov_col)[0]) / 100.0, ls="--", c="black", lw=1)
    axes[2].set_ylabel(cov_col)
    axes[2].set_title("Coverage")
    axes[2].grid(alpha=0.25)

    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(b["label"], rotation=20, ha="right")
        ax.set_xlabel("Band information ladder")

    savefig(fig, out_path, dpi)


def parse_errscale(name: str) -> float | None:
    if name == "baseline":
        return 1.0
    m = re.match(r"^errscale_(\d+(?:\.\d+)?)$", name)
    if not m:
        return None
    return float(m.group(1))


def plot_snr_sweep(df: pd.DataFrame, out_path: str, dpi: int):
    cov_col, width_col = infer_macro_cols(df)

    s = df.copy()
    s["error_factor"] = s["scenario"].map(parse_errscale)
    s = s[s["error_factor"].notnull()].copy()
    if s.empty:
        return
    s["error_factor"] = s["error_factor"].astype(float)
    s["relative_snr"] = 1.0 / s["error_factor"]
    s = s.sort_values("relative_snr")

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.4))

    axes[0].plot(s["relative_snr"], s["rmse_norm_macro"], marker="o", lw=2, color="#264653")
    axes[0].set_ylabel("RMSE (normalized)")
    axes[0].set_title("Accuracy vs Relative SNR")
    axes[0].grid(alpha=0.25)

    axes[1].plot(s["relative_snr"], s[width_col], marker="o", lw=2, color="#2A9D8F")
    axes[1].set_ylabel(width_col)
    axes[1].set_title("Width vs Relative SNR")
    axes[1].grid(alpha=0.25)

    axes[2].plot(s["relative_snr"], s[cov_col], marker="o", lw=2, color="#E76F51")
    axes[2].axhline(float(re.findall(r"\d+", cov_col)[0]) / 100.0, ls="--", c="black", lw=1)
    axes[2].set_ylabel(cov_col)
    axes[2].set_title("Coverage vs Relative SNR")
    axes[2].grid(alpha=0.25)

    for ax in axes:
        ax.set_xlabel("Relative SNR (1 / error_factor)")
        ax.set_xscale("log")

    savefig(fig, out_path, dpi)


def plot_per_param_bands(df: pd.DataFrame, params: list[str], out_path: str, dpi: int):
    cov_col, width_col = infer_param_cols(df)
    m = df["scenario"].eq("baseline") | df["scenario"].str.startswith("survey_")
    b = df[m].copy()
    if b.empty:
        return
    b["order"] = b["scenario"].map(band_order)
    b = b.sort_values(["column", "order"])

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
    for p in params:
        bp = b[b["column"] == p]
        if bp.empty:
            continue
        x = bp["order"].values
        axes[0].plot(x, bp["rmse_norm"], marker="o", lw=1.8, label=p)
        axes[1].plot(x, bp[width_col], marker="o", lw=1.8, label=p)
        axes[2].plot(x, bp[cov_col], marker="o", lw=1.8, label=p)

    x_ticks = [0, 1, 2]
    x_labels = ["gaia only", "gaia+2mass", "full"]
    axes[0].set_title("Per-Parameter RMSE")
    axes[0].set_ylabel("RMSE (normalized)")
    axes[1].set_title(f"Per-Parameter Width ({width_col})")
    axes[1].set_ylabel(width_col)
    axes[2].set_title(f"Per-Parameter Coverage ({cov_col})")
    axes[2].set_ylabel(cov_col)
    for ax in axes:
        ax.set_xticks(x_ticks)
        ax.set_xticklabels(x_labels, rotation=20, ha="right")
        ax.grid(alpha=0.25)
        ax.set_xlabel("Band information")
    axes[2].axhline(float(re.findall(r"\d+", cov_col)[0]) / 100.0, ls="--", c="black", lw=1)
    axes[0].legend(frameon=False, fontsize=8, ncol=2)

    savefig(fig, out_path, dpi)


def plot_per_param_snr(df: pd.DataFrame, params: list[str], out_path: str, dpi: int):
    cov_col, width_col = infer_param_cols(df)
    s = df.copy()
    s["error_factor"] = s["scenario"].map(parse_errscale)
    s = s[s["error_factor"].notnull()].copy()
    if s.empty:
        return
    s["error_factor"] = s["error_factor"].astype(float)
    s["relative_snr"] = 1.0 / s["error_factor"]
    s = s.sort_values(["column", "relative_snr"])

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
    for p in params:
        sp = s[s["column"] == p]
        if sp.empty:
            continue
        axes[0].plot(sp["relative_snr"], sp["rmse_norm"], marker="o", lw=1.8, label=p)
        axes[1].plot(sp["relative_snr"], sp[width_col], marker="o", lw=1.8, label=p)
        axes[2].plot(sp["relative_snr"], sp[cov_col], marker="o", lw=1.8, label=p)

    axes[0].set_title("Per-Parameter RMSE vs Relative SNR")
    axes[0].set_ylabel("RMSE (normalized)")
    axes[1].set_title(f"Per-Parameter Width vs Relative SNR ({width_col})")
    axes[1].set_ylabel(width_col)
    axes[2].set_title(f"Per-Parameter Coverage vs Relative SNR ({cov_col})")
    axes[2].set_ylabel(cov_col)
    for ax in axes:
        ax.grid(alpha=0.25)
        ax.set_xlabel("Relative SNR (1 / error_factor)")
        ax.set_xscale("log")
    axes[2].axhline(float(re.findall(r"\d+", cov_col)[0]) / 100.0, ls="--", c="black", lw=1)
    axes[0].legend(frameon=False, fontsize=8, ncol=2)

    savefig(fig, out_path, dpi)


def main():
    args = parse_args()
    params = [p.strip() for p in args.params.split(",") if p.strip()]
    if not params:
        raise ValueError("Expected at least one parameter name in --params.")

    eval_dir = args.eval_dir
    out_dir = ensure_dir(args.output_dir or os.path.join(eval_dir, "plots"))
    fmt = args.fmt.lower()
    if fmt not in {"png", "pdf"}:
        raise ValueError("--fmt must be one of: png, pdf")

    coverage_file = resolve_file(
        args.coverage_file,
        eval_dir,
        ["coverage_by_level_*.csv"],
    )
    tarp_file = resolve_file(
        args.tarp_file,
        eval_dir,
        ["tarp_curve_*.csv"],
    )
    main_file = resolve_file(
        args.sensitivity_main_file,
        eval_dir,
        ["sensitivity_summary_*main*.csv", "sensitivity_summary_sensitivity_main.csv"],
    )
    bands_file = resolve_file(
        args.sensitivity_bands_file,
        eval_dir,
        ["sensitivity_summary_*bands*.csv", "sensitivity_summary_bands_ladder.csv"],
    )
    snr_file = resolve_file(
        args.sensitivity_snr_file,
        eval_dir,
        ["sensitivity_summary_*snr*.csv", "sensitivity_summary_snr_sweep.csv"],
    )

    per_param_bands_file = first_existing([
        args.per_param_bands_file,
        paired_per_param_path(bands_file),
        latest_by_glob(eval_dir, ["sensitivity_per_param_*bands*.csv", "sensitivity_per_param_bands_ladder.csv"]),
    ])
    per_param_snr_file = first_existing([
        args.per_param_snr_file,
        paired_per_param_path(snr_file),
        latest_by_glob(eval_dir, ["sensitivity_per_param_*snr*.csv", "sensitivity_per_param_snr_sweep.csv"]),
    ])

    manifest = {
        "inputs": {
            "eval_dir": eval_dir,
            "coverage_file": coverage_file,
            "tarp_file": tarp_file,
            "sensitivity_main_file": main_file,
            "sensitivity_bands_file": bands_file,
            "sensitivity_snr_file": snr_file,
            "per_param_bands_file": per_param_bands_file,
            "per_param_snr_file": per_param_snr_file,
        },
        "outputs": [],
    }

    if coverage_file:
        cov_df = pd.read_csv(coverage_file)
        out = os.path.join(out_dir, f"coverage_calibration.{fmt}")
        plot_coverage(cov_df, out, dpi=args.dpi)
        manifest["outputs"].append(out)

    if tarp_file:
        tarp_df = pd.read_csv(tarp_file)
        out = os.path.join(out_dir, f"tarp_calibration.{fmt}")
        plot_tarp(tarp_df, out, dpi=args.dpi)
        manifest["outputs"].append(out)

    frontier_sources = []
    for f in (main_file, bands_file, snr_file):
        if f:
            frontier_sources.append(pd.read_csv(f))
    if frontier_sources:
        frontier_df = pd.concat(frontier_sources, axis=0, ignore_index=True)
        frontier_df = frontier_df.drop_duplicates(subset=["scenario"], keep="last")
        out = os.path.join(out_dir, f"sensitivity_frontier.{fmt}")
        plot_frontier(frontier_df, out, dpi=args.dpi)
        manifest["outputs"].append(out)

    if bands_file:
        bands_df = pd.read_csv(bands_file)
        out = os.path.join(out_dir, f"bands_ladder_macro.{fmt}")
        plot_band_ladder(bands_df, out, dpi=args.dpi)
        manifest["outputs"].append(out)

    if snr_file:
        snr_df = pd.read_csv(snr_file)
        out = os.path.join(out_dir, f"snr_sweep_macro.{fmt}")
        plot_snr_sweep(snr_df, out, dpi=args.dpi)
        manifest["outputs"].append(out)

    if per_param_bands_file:
        pb_df = pd.read_csv(per_param_bands_file)
        out = os.path.join(out_dir, f"bands_ladder_per_param.{fmt}")
        plot_per_param_bands(pb_df, params=params, out_path=out, dpi=args.dpi)
        manifest["outputs"].append(out)

    if per_param_snr_file:
        ps_df = pd.read_csv(per_param_snr_file)
        out = os.path.join(out_dir, f"snr_sweep_per_param.{fmt}")
        plot_per_param_snr(ps_df, params=params, out_path=out, dpi=args.dpi)
        manifest["outputs"].append(out)

    manifest_path = os.path.join(out_dir, "plot_manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"Input eval dir: {eval_dir}")
    print(f"Output plot dir: {out_dir}")
    print("Generated plots:")
    for p in manifest["outputs"]:
        print(f"  - {p}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
