#!/usr/bin/env python3
"""Utilities to visualize checkpoint predictions and errors from evaluation CSVs."""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec


def _sorted_action_columns(df, prefix):
    """Return action columns matching the given prefix in numeric order."""

    cols = [col for col in df.columns if col.startswith(prefix)]
    try:
        return sorted(cols, key=lambda name: int(name.split("_")[1]))
    except (IndexError, ValueError):
        return sorted(cols)


def main(args):
    inp = Path(args.input)
    if not inp.exists():
        sys.exit(f"Input file not found: {inp}")

    df = pd.read_csv(inp)

    if "step" not in df.columns:
        sys.exit("CSV must contain a 'step' column.")

    if "checkpoint" not in df.columns:
        df["checkpoint"] = "checkpoint_0"

    gt_cols = _sorted_action_columns(df, "gt_")
    pred_cols = _sorted_action_columns(df, "pred_")

    if not gt_cols or len(gt_cols) != len(pred_cols):
        sys.exit(
            "CSV must contain matching 'gt_<i>' and 'pred_<i>' columns for each action dimension."
        )

    n_dims = len(gt_cols)

    abs_cols = []
    for idx, (gt_col, pred_col) in enumerate(zip(gt_cols, pred_cols)):
        abs_name = f"abs_err_{idx}"
        df[abs_name] = (df[pred_col] - df[gt_col]).abs()
        abs_cols.append(abs_name)

    group_keys = ["checkpoint", "step"]

    preds_per_ckpt = (
        df.groupby(group_keys)[pred_cols]
        .mean()
        .reset_index()
        .sort_values(group_keys)
    )

    errors_per_ckpt = (
        df.groupby(group_keys)[abs_cols]
        .mean()
        .reset_index()
        .sort_values(group_keys)
    )

    gt_per_step = (
        df.groupby("step")[gt_cols]
        .mean()
        .reset_index()
        .sort_values("step")
    )

    global_max = errors_per_ckpt[abs_cols].to_numpy().max()
    if not np.isfinite(global_max) or global_max <= 0:
        sys.exit("Global maximum error is not positive/finite; cannot normalize.")

    norm_cols = []
    for abs_col in abs_cols:
        norm_name = f"norm_{abs_col}"
        errors_per_ckpt[norm_name] = errors_per_ckpt[abs_col] / global_max
        norm_cols.append(norm_name)

    checkpoints = list(preds_per_ckpt["checkpoint"].unique())
    n_checkpoints = len(checkpoints)

    cmap = plt.get_cmap("tab10", max(n_checkpoints, 1))
    color_map = {ckpt: cmap(idx % cmap.N) for idx, ckpt in enumerate(checkpoints)}

    fig_width = max(12.0, 4.5 * n_dims)
    fig_height = 10.5
    fig = plt.figure(figsize=(fig_width, fig_height))
    grid = GridSpec(3, n_dims, figure=fig, height_ratios=[2.5, 1.0, 1.0], hspace=0.4, wspace=0.35)

    pred_axes = [fig.add_subplot(grid[0, dim_idx]) for dim_idx in range(n_dims)]
    raw_axes = [fig.add_subplot(grid[1, dim_idx]) for dim_idx in range(n_dims)]
    norm_axes = [fig.add_subplot(grid[2, dim_idx]) for dim_idx in range(n_dims)]

    # Row 1: predictions vs ground truth
    for dim_idx, (gt_col, pred_col) in enumerate(zip(gt_cols, pred_cols)):
        ax = pred_axes[dim_idx]

        for ckpt in checkpoints:
            ckpt_rows = preds_per_ckpt[preds_per_ckpt["checkpoint"] == ckpt]
            ax.plot(
                ckpt_rows["step"],
                ckpt_rows[pred_col],
                label=ckpt if dim_idx == 0 else None,
                color=color_map[ckpt],
                alpha=0.85,
            )

        ax.plot(
            gt_per_step["step"],
            gt_per_step[gt_col],
            label="ground truth" if dim_idx == 0 else None,
            color="black",
            linewidth=2.0,
            linestyle="--",
        )

        ax.set_title(f"Dim {dim_idx}: prediction vs GT")
        ax.set_ylabel("Value")
        ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.4)

    # Row 2: absolute error
    for dim_idx, abs_col in enumerate(abs_cols):
        ax = raw_axes[dim_idx]

        for ckpt in checkpoints:
            ckpt_rows = errors_per_ckpt[errors_per_ckpt["checkpoint"] == ckpt]
            ax.plot(
                ckpt_rows["step"],
                ckpt_rows[abs_col],
                label=ckpt if dim_idx == 0 else None,
                color=color_map[ckpt],
                alpha=0.85,
            )

        ax.set_ylabel("|gt - pred|")
        ax.set_title(f"Dim {dim_idx}: absolute error")
        ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.4)

    # Row 3: normalized absolute error
    for dim_idx, norm_col in enumerate(norm_cols):
        ax = norm_axes[dim_idx]

        for ckpt in checkpoints:
            ckpt_rows = errors_per_ckpt[errors_per_ckpt["checkpoint"] == ckpt]
            ax.plot(
                ckpt_rows["step"],
                ckpt_rows[norm_col],
                label=ckpt if dim_idx == 0 else None,
                color=color_map[ckpt],
                alpha=0.85,
            )

        ax.set_xlabel("Step")
        ax.set_ylabel("Normalized |gt - pred|")
        ax.set_title(f"Dim {dim_idx}: normalized error")
        ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.4)

    fig.suptitle(f"Predictions and errors per checkpoint ({inp.name})", fontsize=14)
    fig.tight_layout(rect=[0, 0, 0.85, 0.94])

    legend_handles, legend_labels = pred_axes[0].get_legend_handles_labels()
    if legend_handles:
        fig.legend(
            legend_handles,
            legend_labels,
            loc="center left",
            bbox_to_anchor=(0.99, 0.5),
            frameon=False,
        )

    out_png = Path(args.out_png)
    plt.savefig(out_png, dpi=160)
    plt.close(fig)

    print(f"Saved chart: {out_png}")
    print(f"Global max (pre-normalization): {global_max:.6f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default=r"C:\Users\sa-forest\Documents\GitHub\robomimic\bc_patcherBot\PipetteFinding\results\v0_009\results_bc_PatcherBot_v0_009_5.csv",
        help="Path to input CSV",
    )
    parser.add_argument(
        "--out-png",
        default=r"C:\Users\sa-forest\Documents\GitHub\robomimic\bc_patcherBot\PipetteFinding\results\results_bc_PatcherBot_v0_009_5_preds_and_errors.png",
        help="Output PNG path",
    )
    args = parser.parse_args()
    main(args)
