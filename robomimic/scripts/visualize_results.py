#!/usr/bin/env python3
"""Utilities to visualize checkpoint predictions and errors from evaluation CSVs."""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec



# Frequency bands used to highlight ground-truth spectra in the FFT figure.
FFT_BAND_RATIOS = [
    ("low", 0.0, 0.02),
    ("mid", 0.2, 0.5),
    ("high", 0.5, 1.0),
]

def _sorted_action_columns(df, prefix):
    """Return action columns matching the given prefix in numeric order."""

    cols = [col for col in df.columns if col.startswith(prefix)]
    try:
        return sorted(cols, key=lambda name: int(name.split("_")[1]))
    except (IndexError, ValueError):
        return sorted(cols)


def _generate_fft_plot(
    gt_per_step,
    preds_per_ckpt,
    gt_cols,
    pred_cols,
    checkpoints,
    color_map,
    out_path,
    source_name,
):
    """Render FFT magnitude plots per action dimension and save to disk."""

    gt_rows = gt_per_step.sort_values("step")
    steps = gt_rows["step"].to_numpy()
    if steps.size < 2:
        print("Skipping FFT plot: need at least two timesteps for FFT.")
        return

    step_diffs = np.diff(steps)
    positive_diffs = step_diffs[step_diffs > 0]
    step_spacing = float(positive_diffs.mean()) if positive_diffs.size else 1.0

    n_dims = len(gt_cols)
    fig_width = max(12.0, 4.5 * n_dims)
    fig_height = 4.5
    fig = plt.figure(figsize=(fig_width, fig_height))
    axes = [fig.add_subplot(1, n_dims, dim_idx + 1) for dim_idx in range(n_dims)]

    band_colors = []
    if FFT_BAND_RATIOS:
        band_cmap = plt.get_cmap("Set2", len(FFT_BAND_RATIOS))
        band_colors = [band_cmap(i) for i in range(band_cmap.N)]

    for dim_idx, (ax, gt_col, pred_col) in enumerate(zip(axes, gt_cols, pred_cols)):
        gt_signal = gt_rows[gt_col].to_numpy()
        if gt_signal.size < 2:
            ax.text(
                0.5,
                0.5,
                "Insufficient data",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            ax.set_axis_off()
            continue

        freqs = np.fft.rfftfreq(
            gt_signal.size, d=step_spacing if step_spacing > 0 else 1.0
        )
        gt_fft = np.abs(np.fft.rfft(gt_signal - gt_signal.mean()))
        ax.plot(
            freqs,
            gt_fft,
            label="ground truth" if dim_idx == 0 else None,
            color="black",
            linewidth=2.0,
            linestyle="--",
        )

        if band_colors and freqs.size:
            max_freq = float(freqs.max())
            if max_freq > 0.0:
                for band_idx, (band_name, lower_ratio, upper_ratio) in enumerate(FFT_BAND_RATIOS):
                    upper_ratio_val = 1.0 if upper_ratio is None else upper_ratio
                    lower = float(max_freq * max(lower_ratio, 0.0))
                    upper = float(max_freq * max(min(upper_ratio_val, 1.0), 0.0))
                    if upper <= lower:
                        continue
                    if np.isclose(upper, max_freq):
                        mask = (freqs >= lower) & (freqs <= upper)
                    else:
                        mask = (freqs >= lower) & (freqs < upper)
                    if not np.any(mask):
                        continue
                    band_fft = np.where(mask, gt_fft, np.nan)
                    legend_label = (
                        f"GT {band_name} [{lower:.3g}, {upper:.3g}] 1/step"
                        if dim_idx == 0
                        else None
                    )
                    ax.plot(
                        freqs,
                        band_fft,
                        label=legend_label,
                        color=band_colors[band_idx % len(band_colors)],
                        linewidth=1.5,
                        alpha=0.8,
                    )

        # Align prediction series to ground-truth steps with interpolation.
        for ckpt in checkpoints:
            ckpt_rows = (
                preds_per_ckpt[preds_per_ckpt["checkpoint"] == ckpt]
                .sort_values("step")
            )
            merged = pd.DataFrame({"step": gt_rows["step"]}).merge(
                ckpt_rows[["step", pred_col]], on="step", how="left"
            )
            series = merged[pred_col].interpolate(limit_direction="both")
            series = series.fillna(method="ffill").fillna(method="bfill")
            if series.isna().all():
                continue

            pred_signal = series.to_numpy()
            if pred_signal.size != gt_signal.size:
                continue

            pred_fft = np.abs(np.fft.rfft(pred_signal - pred_signal.mean()))
            ax.plot(
                freqs,
                pred_fft,
                label=ckpt if dim_idx == 0 else None,
                color=color_map[ckpt],
                alpha=0.85,
            )

        ax.set_title(f"Dim {dim_idx}: FFT magnitude")
        ax.set_xlabel("Frequency (1/step)")
        ax.set_ylabel("Magnitude")
        ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.4)

    fig.suptitle(f"FFT magnitude per checkpoint ({source_name})", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1.0, 0.92])

    if axes:
        legend_handles, legend_labels = axes[0].get_legend_handles_labels()
        if legend_handles:
            fig.legend(
                legend_handles,
                legend_labels,
                loc="upper center",
                bbox_to_anchor=(0.5, 0.98),
                frameon=False,
                ncol=1,
            )

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)

    print(f"Saved FFT chart: {out_path}")


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
    fig.tight_layout(rect=[0, 0, 1.0, 0.9])

    legend_handles, legend_labels = pred_axes[0].get_legend_handles_labels()
    if legend_handles:
        fig.legend(
            legend_handles,
            legend_labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.98),
            frameon=False,
            ncol=1,
        )

    out_png = Path(args.out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=160)
    plt.close(fig)

    print(f"Saved chart: {out_png}")
    print(f"Global max (pre-normalization): {global_max:.6f}")

    if args.out_fft_png:
        _generate_fft_plot(
            gt_per_step=gt_per_step,
            preds_per_ckpt=preds_per_ckpt,
            gt_cols=gt_cols,
            pred_cols=pred_cols,
            checkpoints=checkpoints,
            color_map=color_map,
            out_path=args.out_fft_png,
            source_name=inp.name,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default=r"C:\Users\sa-forest\Documents\GitHub\robomimic\bc_patcherBot\PipetteFinding\results\v0_130\results_bc_PatcherBot_v0_130_0.csv",
        help="Path to input CSV",
    )
    parser.add_argument(
        "--out-png",
        default=r"C:\Users\sa-forest\Documents\GitHub\robomimic\bc_patcherBot\PipetteFinding\results\v0_130\results_bc_PatcherBot_v0_130_0_preds_and_errors.png",
        help="Output PNG path",
    )
    parser.add_argument(
        "--out-fft-png",
        default=r"C:\Users\sa-forest\Documents\GitHub\robomimic\bc_patcherBot\PipetteFinding\results\v0_130\results_bc_PatcherBot_v0_130_0_fft.png",
        help="Optional output PNG path for FFT plot; if omitted the FFT figure is skipped.",
    )
    args = parser.parse_args()
    main(args)
