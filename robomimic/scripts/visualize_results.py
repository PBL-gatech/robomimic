#!/usr/bin/env python3
"""
Compute and plot per-step absolute errors for 6 predictions, both raw and normalized.

Expected columns in the CSV:
  step, gt_0..gt_5, pred_0..pred_5

Outputs:
  - <out_png>: 2x1 figure
        Top:   abs_err_0..abs_err_5 (mean per step, non-normalized)
        Bottom: norm_abs_err_0..norm_abs_err_5 (normalized to global max across series)
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import sys

def main(args):
    inp = Path(args.input)
    if not inp.exists():
        sys.exit(f"Input file not found: {inp}")

    df = pd.read_csv(inp)

    # Basic checks
    if "step" not in df.columns:
        sys.exit("CSV must contain a 'step' column.")

    pairs = []
    for i in range(6):
        g, p = f"gt_{i}", f"pred_{i}"
        if g not in df.columns or p not in df.columns:
            sys.exit(f"Missing required columns: '{g}' and/or '{p}'")
        pairs.append((g, p))

    # Compute abs error columns (row-level)
    for i, (g, p) in enumerate(pairs):
        df[f"abs_err_{i}"] = (df[p] - df[g]).abs()

    # Aggregate by step (mean per step)
    per_step = (
        df.groupby("step")[[f"abs_err_{i}" for i in range(6)]]
        .mean()
        .reset_index()
        .sort_values("step")
    )

    # Normalize by the single largest per-step error across all series
    global_max = per_step[[f"abs_err_{i}" for i in range(6)]].to_numpy().max()
    if not np.isfinite(global_max) or global_max <= 0:
        sys.exit("Global maximum error is not positive/finite; cannot normalize.")

    for i in range(6):
        per_step[f"norm_abs_err_{i}"] = per_step[f"abs_err_{i}"] / global_max

    # Prepare data for plotting
    raw_cols = [f"abs_err_{i}" for i in range(6)]
    norm_cols = [f"norm_abs_err_{i}" for i in range(6)]

    # Plot: 2x1 (top: raw, bottom: normalized)
    fig, axes = plt.subplots(2, 1, figsize=(8, 8), sharex=True)

    # Top: raw absolute errors
    ax_raw = axes[0]
    for i, col in enumerate(raw_cols):
        ax_raw.plot(per_step["step"], per_step[col], label=col)
    ax_raw.set_ylabel("|gt - pred|")
    ax_raw.set_title("Per-step absolute error (6 predictions)")
    ax_raw.legend(loc="best")
    ax_raw.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)

    # Bottom: normalized absolute errors
    ax_norm = axes[1]
    for i, col in enumerate(norm_cols):
        ax_norm.plot(per_step["step"], per_step[col], label=col)
    ax_norm.set_xlabel("step")
    ax_norm.set_ylabel("Normalized |gt - pred| (max = 1)")
    ax_norm.set_title("Per-step absolute error (normalized)")
    ax_norm.legend(loc="best")
    ax_norm.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)

    fig.tight_layout()

    out_png = Path(args.out_png)
    plt.savefig(out_png, dpi=160)
    plt.close()

    print(f"Saved chart: {out_png}")
    print(f"Global max (pre-normalization): {global_max:.6f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default=r"C:\Users\sa-forest\Documents\GitHub\robomimic\df_patcherBot\PipetteFinding\results\results_df_PatcherBot_v0_003.csv",
        help="Path to input CSV",
    )
    parser.add_argument(
        "--out-png",
        # removed 'norm' from the filename and clarified that both views are included
        default=r"C:\Users\sa-forest\Documents\GitHub\robomimic\df_patcherBot\PipetteFinding\results\results_df_PatcherBot_v0_003_abs_error_and_normalized_all6_per_step.png",
        help="Output PNG path",
    )
    args = parser.parse_args()
    main(args)
