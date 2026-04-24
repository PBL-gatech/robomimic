"""Shared plotting utilities for PatcherBot evaluation outputs."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec


FFT_BAND_RATIOS = [
    ("low", 0.0, 0.02),
    ("mid", 0.2, 0.5),
    ("high", 0.5, 1.0),
]


def _sorted_action_columns(df: pd.DataFrame, prefix: str) -> List[str]:
    cols = [col for col in df.columns if col.startswith(prefix)]
    try:
        return sorted(cols, key=lambda name: int(name.split("_")[1]))
    except (IndexError, ValueError):
        return sorted(cols)


def default_plot_paths_for_csv(csv_path: str, plot_dir: str) -> Tuple[Path, Path]:
    csv_file = Path(csv_path).expanduser()
    plot_root = Path(plot_dir).expanduser()
    ckpt_stem = csv_file.stem.replace("results_bc_PatcherBot_", "")
    return (
        plot_root / f"predictions_{ckpt_stem}.png",
        plot_root / f"fft_{ckpt_stem}.png",
    )


def _generate_fft_plot(
    gt_per_step: pd.DataFrame,
    preds_per_ckpt: pd.DataFrame,
    gt_cols: List[str],
    pred_cols: List[str],
    checkpoints: List[str],
    color_map: Dict[str, Any],
    out_path: Path,
    source_name: str,
) -> None:
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
    fig = plt.figure(figsize=(fig_width, 4.5))
    axes = [fig.add_subplot(1, n_dims, dim_idx + 1) for dim_idx in range(n_dims)]

    band_colors = []
    if FFT_BAND_RATIOS:
        band_cmap = plt.get_cmap("Set2", len(FFT_BAND_RATIOS))
        band_colors = [band_cmap(i) for i in range(band_cmap.N)]

    mse_per_checkpoint = {ckpt: {} for ckpt in checkpoints}

    for dim_idx, (ax, gt_col, pred_col) in enumerate(zip(axes, gt_cols, pred_cols)):
        gt_signal = gt_rows[gt_col].to_numpy()
        if gt_signal.size < 2:
            ax.text(0.5, 0.5, "Insufficient data", ha="center", va="center", transform=ax.transAxes)
            ax.set_axis_off()
            continue

        freqs = np.fft.rfftfreq(gt_signal.size, d=step_spacing if step_spacing > 0 else 1.0)
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
                    ax.plot(
                        freqs,
                        band_fft,
                        label=f"GT {band_name}" if dim_idx == 0 else None,
                        color=band_colors[band_idx % len(band_colors)],
                        linewidth=1.5,
                        alpha=0.8,
                    )

        for ckpt in checkpoints:
            ckpt_rows = preds_per_ckpt[preds_per_ckpt["checkpoint"] == ckpt].sort_values("step")
            merged = pd.DataFrame({"step": gt_rows["step"]}).merge(
                ckpt_rows[["step", pred_col]],
                on="step",
                how="left",
            )
            series = merged[pred_col].interpolate(limit_direction="both")
            series = series.ffill().bfill()
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

            diff = pred_fft - gt_fft
            mse_val = float(np.mean(np.square(diff)))
            if np.isfinite(mse_val):
                mse_per_checkpoint[ckpt][dim_idx] = mse_val

        ax.set_title(f"Dim {dim_idx}: FFT magnitude")
        ax.set_xlabel("Frequency (1/step)")
        ax.set_ylabel("Magnitude")
        ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.4)

    if any(dim_mses for dim_mses in mse_per_checkpoint.values()):
        print("FFT magnitude MSE (prediction vs ground truth):")
        for ckpt in checkpoints:
            dim_mses = mse_per_checkpoint.get(ckpt, {})
            if not dim_mses:
                continue
            mean_mse = float(np.mean(list(dim_mses.values())))
            dim_entries = ", ".join(f"dim{dim_idx}: {mse:.6g}" for dim_idx, mse in sorted(dim_mses.items()))
            print(f"  {ckpt}: mean={mean_mse:.6g} ({dim_entries})")

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

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    print(f"Saved FFT chart: {out_path}")


def generate_prediction_plots(
    csv_path: str,
    out_png_path: str,
    out_fft_png_path: Optional[str] = None,
) -> Dict[str, Any]:
    csv_file = Path(csv_path).expanduser()
    if not csv_file.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_file}")

    df = pd.read_csv(csv_file)
    if "step" not in df.columns:
        raise ValueError("CSV must contain a 'step' column.")

    if "checkpoint" not in df.columns:
        df["checkpoint"] = "checkpoint_0"

    gt_cols = _sorted_action_columns(df, "gt_")
    pred_cols = _sorted_action_columns(df, "pred_")
    if not gt_cols or len(gt_cols) != len(pred_cols):
        raise ValueError("CSV must contain matching 'gt_<i>' and 'pred_<i>' columns for each action dimension.")

    abs_cols = []
    for idx, (gt_col, pred_col) in enumerate(zip(gt_cols, pred_cols)):
        abs_name = f"abs_err_{idx}"
        df[abs_name] = (df[pred_col] - df[gt_col]).abs()
        abs_cols.append(abs_name)

    group_keys = ["checkpoint", "step"]
    preds_per_ckpt = df.groupby(group_keys)[pred_cols].mean().reset_index().sort_values(group_keys)
    errors_per_ckpt = df.groupby(group_keys)[abs_cols].mean().reset_index().sort_values(group_keys)
    gt_per_step = df.groupby("step")[gt_cols].mean().reset_index().sort_values("step")

    checkpoints = list(preds_per_ckpt["checkpoint"].unique())
    n_dims = len(gt_cols)
    cmap = plt.get_cmap("tab10", max(len(checkpoints), 1))
    color_map = {ckpt: cmap(idx % cmap.N) for idx, ckpt in enumerate(checkpoints)}

    fig_width = max(12.0, 4.5 * n_dims)
    fig = plt.figure(figsize=(fig_width, 10.5))
    grid = GridSpec(3, n_dims, figure=fig, height_ratios=[2.5, 1.0, 1.0], hspace=0.4, wspace=0.35)

    pred_axes = [fig.add_subplot(grid[0, dim_idx]) for dim_idx in range(n_dims)]
    raw_axes = [fig.add_subplot(grid[1, dim_idx]) for dim_idx in range(n_dims)]
    norm_axes = [fig.add_subplot(grid[2, dim_idx]) for dim_idx in range(n_dims)]

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

    global_max = errors_per_ckpt[abs_cols].to_numpy().max()
    if not np.isfinite(global_max) or global_max <= 0:
        global_max = 1.0

    norm_cols = []
    for abs_col in abs_cols:
        norm_name = f"norm_{abs_col}"
        errors_per_ckpt[norm_name] = errors_per_ckpt[abs_col] / global_max
        norm_cols.append(norm_name)

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

    fig.suptitle(f"Predictions and errors per checkpoint ({csv_file.name})", fontsize=14)
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

    output_png = Path(out_png_path).expanduser()
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=160)
    plt.close(fig)

    fft_output = None
    if out_fft_png_path:
        fft_output = Path(out_fft_png_path).expanduser()
        _generate_fft_plot(
            gt_per_step=gt_per_step,
            preds_per_ckpt=preds_per_ckpt,
            gt_cols=gt_cols,
            pred_cols=pred_cols,
            checkpoints=checkpoints,
            color_map=color_map,
            out_path=fft_output,
            source_name=csv_file.name,
        )

    print(f"Saved plot: {output_png}")
    print(f"Global max (pre-normalization): {global_max:.6f}")

    return {
        "csv_path": str(csv_file),
        "output_png": str(output_png),
        "fft_png": str(fft_output) if fft_output is not None else None,
        "global_max": float(global_max),
        "checkpoints": checkpoints,
        "n_dims": n_dims,
    }


def batch_generate_plots(csv_dir: str, plot_dir: str, generate_fft: bool = True) -> List[Dict[str, Any]]:
    csv_root = Path(csv_dir).expanduser()
    plot_root = Path(plot_dir).expanduser()
    if not csv_root.exists():
        raise FileNotFoundError(f"CSV directory not found: {csv_root}")

    csv_files = sorted(csv_root.glob("results_bc_PatcherBot_*.csv"))
    if not csv_files:
        print(f"No CSV files found in {csv_root}")
        return []

    results = []
    for csv_path in csv_files:
        try:
            out_png, out_fft = default_plot_paths_for_csv(str(csv_path), str(plot_root))
            plot_result = generate_prediction_plots(
                csv_path=str(csv_path),
                out_png_path=str(out_png),
                out_fft_png_path=str(out_fft) if generate_fft else None,
            )
            results.append(plot_result)
        except Exception as exc:
            print(f"[ERROR] Failed to generate plots for {csv_path.name}: {exc}")
            results.append(
                {
                    "csv_path": str(csv_path),
                    "error": str(exc),
                }
            )
    return results


__all__ = [
    "FFT_BAND_RATIOS",
    "batch_generate_plots",
    "default_plot_paths_for_csv",
    "generate_prediction_plots",
]
