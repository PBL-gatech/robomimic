#!/usr/bin/env python3
"""
Visualize per-sequence diffusion predictions produced by
scripts/run_PatcherBot_diffusionAgent.py.

Inputs:
 - per-element CSV (required): each row corresponds to one sequence element with columns:
     checkpoint, seq_index, elem, t, pred_<i>, gt_<i>, err_<i>
 - per-sequence CSV (optional): one row per sequence with summary metrics

This tool renders, for selected sequences:
 - Per-dimension predicted vs ground-truth trajectories across the action horizon (elem axis)
 - Per-element error norm across dimensions
 - Error heatmap (dim x elem)
"""

import argparse
import sys
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _sorted_action_columns(df: pd.DataFrame, prefix: str) -> List[str]:
    cols = [c for c in df.columns if c.startswith(prefix)]
    try:
        return sorted(cols, key=lambda name: int(name.split("_")[1]))
    except (IndexError, ValueError):
        return sorted(cols)


def _select_sequences(
    df_elem: pd.DataFrame,
    checkpoint_filter: str,
    seq_indices: List[int],
    topk: int,
    pred_cols: List[str],
    gt_cols: List[str],
):
    data = df_elem
    if checkpoint_filter:
        data = data[data["checkpoint"].astype(str).str.contains(checkpoint_filter)]
        if data.empty:
            sys.exit(f"No rows match checkpoint filter: {checkpoint_filter}")

    # Aggregate sequence-level RMSE from per-element details
    def _agg(g: pd.DataFrame):
        pred = g[pred_cols].to_numpy()
        gt = g[gt_cols].to_numpy()
        diff = pred - gt
        mse = float(np.mean(np.square(diff)))
        rmse = float(np.sqrt(max(np.mean(np.square(diff)), 0.0)))
        mae = float(np.mean(np.abs(diff)))
        return pd.Series({"rmse": rmse, "mae": mae, "n": len(g)})

    summary = (
        data.groupby(["checkpoint", "seq_index"], as_index=False)
        .apply(_agg)
        .reset_index(drop=True)
        .sort_values(["rmse"], ascending=False)
    )

    if seq_indices:
        wanted = (
            summary[summary["seq_index"].isin(seq_indices)]
            .sort_values(["checkpoint", "seq_index"])
        )
    else:
        wanted = summary.groupby("checkpoint").head(topk)

    if wanted.empty:
        sys.exit("No sequences selected for plotting.")
    return wanted


def _plot_one_sequence(
    g: pd.DataFrame,
    pred_cols: List[str],
    gt_cols: List[str],
    dims_to_plot: List[int],
    title_prefix: str,
    out_path: Path,
):
    # Ensure ordering by element
    g = g.sort_values(["elem"])  # elem in [0, Ta)

    # Prepare arrays
    X = g["elem"].to_numpy()
    P = g[pred_cols].to_numpy()
    G = g[gt_cols].to_numpy()
    E = np.abs(P - G)

    D = P.shape[1]
    dims = [d for d in dims_to_plot if 0 <= d < D]
    if not dims:
        dims = list(range(min(D, 6)))

    # Compute global max abs error for normalization (for this sequence)
    global_max = float(np.nanmax(E)) if E.size else 0.0
    if not np.isfinite(global_max) or global_max <= 0:
        global_max = 1.0

    n_dims = len(dims)
    fig_width = max(12.0, 4.5 * n_dims)
    fig_height = 10.5
    fig = plt.figure(figsize=(fig_width, fig_height))

    from matplotlib.gridspec import GridSpec
    grid = GridSpec(3, n_dims, figure=fig, height_ratios=[2.5, 1.0, 1.0], hspace=0.4, wspace=0.35)

    pred_axes = [fig.add_subplot(grid[0, j]) for j in range(n_dims)]
    raw_axes = [fig.add_subplot(grid[1, j]) for j in range(n_dims)]
    norm_axes = [fig.add_subplot(grid[2, j]) for j in range(n_dims)]

    # Row 1: predictions vs ground truth, per dim
    for ax, d in zip(pred_axes, dims):
        ax.plot(X, P[:, d], label="pred", color="#1f77b4", alpha=0.9)
        ax.plot(X, G[:, d], label="ground truth", color="black", linewidth=2.0, linestyle="--")
        ax.set_title(f"Dim {d}: prediction vs GT")
        ax.set_ylabel("Value")
        ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.4)

    # Row 2: absolute error per dim
    for ax, d in zip(raw_axes, dims):
        abs_err = np.abs(P[:, d] - G[:, d])
        ax.plot(X, abs_err, color="#d62728", alpha=0.9)
        ax.set_title(f"Dim {d}: absolute error")
        ax.set_ylabel("|gt - pred|")
        ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.4)

    # Row 3: normalized absolute error per dim
    for ax, d in zip(norm_axes, dims):
        abs_err = np.abs(P[:, d] - G[:, d])
        norm_err = abs_err / global_max
        ax.plot(X, norm_err, color="#ff7f0e", alpha=0.9)
        ax.set_xlabel("Elem (within Ta)")
        ax.set_title(f"Dim {d}: normalized error")
        ax.set_ylabel("Normalized |gt - pred|")
        ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.4)

    # Put a single legend at the top from the first pred axis
    handles, labels = pred_axes[0].get_legend_handles_labels() if pred_axes else ([], [])
    if handles:
        fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.98), frameon=False, ncol=2)

    fig.suptitle(title_prefix, fontsize=14)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=[0, 0, 1.0, 0.92])
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    print(f"Saved: {out_path}")


def _plot_fft_sequence(
    g: pd.DataFrame,
    pred_cols: List[str],
    gt_cols: List[str],
    dims_to_plot: List[int],
    title_prefix: str,
    out_path: Path,
):
    # Order by elem within sequence
    g = g.sort_values(["elem"])  # elem in [0, Ta)

    X = g["elem"].to_numpy()
    P = g[pred_cols].to_numpy()
    G = g[gt_cols].to_numpy()

    D = P.shape[1]
    dims = [d for d in dims_to_plot if 0 <= d < D]
    if not dims:
        dims = list(range(min(D, 6)))

    # Treat elem index spacing as 1.0
    d_step = 1.0

    n_dims = len(dims)
    fig_width = max(12.0, 4.5 * n_dims)
    fig_height = 4.5
    fig = plt.figure(figsize=(fig_width, fig_height))
    axes = [fig.add_subplot(1, n_dims, i + 1) for i in range(n_dims)]

    for ax, d in zip(axes, dims):
        gt_signal = G[:, d]
        pred_signal = P[:, d]
        if gt_signal.size < 2:
            ax.text(0.5, 0.5, "Insufficient data", ha="center", va="center", transform=ax.transAxes)
            ax.set_axis_off()
            continue

        freqs = np.fft.rfftfreq(gt_signal.size, d=d_step)
        gt_fft = np.abs(np.fft.rfft(gt_signal - float(np.mean(gt_signal))))
        pr_fft = np.abs(np.fft.rfft(pred_signal - float(np.mean(pred_signal))))

        ax.plot(freqs, gt_fft, label="ground truth", color="black", linewidth=2.0, linestyle="--")
        ax.plot(freqs, pr_fft, label="pred", color="#1f77b4", alpha=0.9)
        ax.set_title(f"Dim {d}: FFT magnitude")
        ax.set_xlabel("Frequency (1/elem)")
        ax.set_ylabel("Magnitude")
        ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.4)

    # Legend once
    if axes:
        handles, labels = axes[0].get_legend_handles_labels()
        if handles:
            fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.98), frameon=False, ncol=2)

    fig.suptitle(title_prefix + " | FFT", fontsize=14)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=[0, 0, 1.0, 0.92])
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    print(f"Saved: {out_path}")


def main(args):
    elem_csv = Path(args.input)
    if not elem_csv.exists():
        sys.exit(f"Per-element CSV not found: {elem_csv}")

    df = pd.read_csv(elem_csv)

    # Basic sanity checks
    need_cols = {"checkpoint", "seq_index", "elem"}
    if not need_cols.issubset(set(df.columns)):
        sys.exit("CSV must contain columns: 'checkpoint', 'seq_index', 'elem'")

    pred_cols = _sorted_action_columns(df, "pred_")
    gt_cols = _sorted_action_columns(df, "gt_")
    if not pred_cols or len(pred_cols) != len(gt_cols):
        sys.exit("CSV must have matching 'pred_<i>' and 'gt_<i>' columns.")

    # Select sequences
    selected = _select_sequences(
        df_elem=df,
        checkpoint_filter=args.checkpoint,
        seq_indices=args.seq_index or [],
        topk=args.topk,
        pred_cols=pred_cols,
        gt_cols=gt_cols,
    )

    dims_to_plot = args.dims if args.dims is not None else list(range(min(len(pred_cols), 6)))
    out_dir = Path(args.out_dir)

    # Plot per selected sequence
    for _, row in selected.iterrows():
        ckpt = str(row["checkpoint"])
        sid = int(row["seq_index"])
        g = df[(df["checkpoint"] == ckpt) & (df["seq_index"] == sid)]
        if g.empty:
            continue
        title = f"{ckpt} | seq {sid} | rmse={row['rmse']:.6f} mae={row['mae']:.6f}"
        safe_ckpt = ckpt.replace('/', '_').replace('\\', '_')
        fname = f"traj_{safe_ckpt}_seq_{sid}.png"
        out_path = out_dir / fname
        _plot_one_sequence(
            g=g,
            pred_cols=pred_cols,
            gt_cols=gt_cols,
            dims_to_plot=dims_to_plot,
            title_prefix=title,
            out_path=out_path,
        )
        if args.fft:
            fft_out = out_dir / f"traj_{safe_ckpt}_seq_{sid}_fft.png"
            _plot_fft_sequence(
                g=g,
                pred_cols=pred_cols,
                gt_cols=gt_cols,
                dims_to_plot=dims_to_plot,
                title_prefix=title,
                out_path=fft_out,
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default=r"C:\\Users\\sa-forest\\Documents\\GitHub\\robomimic\\df_patcherBot\\PipetteFinding\\results\\v0_160\\results_df_PatcherBot_traj_v0_160_0.csv",
        help="Path to per-element sequence CSV (from run_PatcherBot_diffusionAgent.py --seq_elements_csv)",
    )
    parser.add_argument(
        "--checkpoint",
        default="",
        help="Optional substring to filter checkpoint names",
    )
    parser.add_argument(
        "--seq-index",
        nargs="*",
        type=int,
        default=None,
        help="Optional list of specific sequence indices to visualize",
    )
    parser.add_argument(
        "--topk",
        type=int,
        default=4,
        help="If no seq-index is provided, plot top-K sequences by RMSE per checkpoint",
    )
    parser.add_argument(
        "--dims",
        nargs="*",
        type=int,
        default=None,
        help="Action dimensions to plot (defaults to first 6 dims)",
    )
    parser.add_argument(
        "--out-dir",
        default=r"C:\\Users\\sa-forest\\Documents\\GitHub\\robomimic\\df_patcherBot\\PipetteFinding\\results\\v0_160",
        help="Output directory to save figures",
    )
    parser.add_argument(
        "--fft",
        default= True,
        action="store_true",
        help="Also save per-sequence FFT magnitude plots",
    )
    args = parser.parse_args()
    main(args)
