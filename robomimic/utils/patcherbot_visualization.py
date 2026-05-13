"""Shared plotting utilities for PatcherBot evaluation outputs."""

from __future__ import annotations

import json
from datetime import datetime
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
    indexed_cols = []
    for col in df.columns:
        if not col.startswith(prefix):
            continue
        suffix = col[len(prefix):]
        try:
            indexed_cols.append((int(suffix), col))
        except ValueError:
            continue
    return [col for _, col in sorted(indexed_cols)]


def default_plot_paths_for_csv(csv_path: str, plot_dir: str) -> Tuple[Path, Path]:
    csv_file = Path(csv_path).expanduser()
    plot_root = Path(plot_dir).expanduser()
    ckpt_stem = csv_file.stem.replace("results_bc_PatcherBot_", "")
    return (
        plot_root / f"predictions_{ckpt_stem}.png",
        plot_root / f"fft_{ckpt_stem}.png",
    )


def _safe_ratio(numerator: float, denominator: float) -> Optional[float]:
    if denominator <= 0:
        return None
    return float(numerator) / float(denominator)


def _run_lengths(mask: np.ndarray) -> List[int]:
    values = np.asarray(mask, dtype=bool).reshape(-1)
    lengths: List[int] = []
    current = 0
    for value in values:
        if value:
            current += 1
        elif current:
            lengths.append(current)
            current = 0
    if current:
        lengths.append(current)
    return lengths


def _is_binary_like(*arrays: np.ndarray) -> bool:
    finite_values = []
    for array in arrays:
        values = np.asarray(array, dtype=np.float64).reshape(-1)
        values = values[np.isfinite(values)]
        if values.size:
            finite_values.append(values)
    if not finite_values:
        return False
    values = np.concatenate(finite_values)
    return bool(np.all(np.isclose(values, 0.0, atol=1e-6) | np.isclose(values, 1.0, atol=1e-6)))


def _transition_indices(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=bool).reshape(-1)
    if arr.size < 2:
        return np.asarray([], dtype=int)
    return np.flatnonzero(arr[1:] != arr[:-1]) + 1


def _nearest_transition_error(gt_edges: np.ndarray, pred_edges: np.ndarray) -> Optional[float]:
    if gt_edges.size == 0 or pred_edges.size == 0:
        return None
    errors = [float(np.min(np.abs(pred_edges - edge))) for edge in gt_edges]
    return float(np.mean(errors)) if errors else None


def _series_metrics(
    *,
    source_csv: str,
    csv_path: str,
    checkpoint: str,
    demo_key: Any,
    demo_id: Any,
    dim_idx: int,
    gt_values: np.ndarray,
    pred_values: np.ndarray,
) -> Dict[str, Any]:
    gt = np.asarray(gt_values, dtype=np.float64).reshape(-1)
    pred = np.asarray(pred_values, dtype=np.float64).reshape(-1)
    valid = np.isfinite(gt) & np.isfinite(pred)
    gt = gt[valid]
    pred = pred[valid]

    row: Dict[str, Any] = {
        "source_csv": source_csv,
        "csv_path": csv_path,
        "checkpoint": checkpoint,
        "demo_key": "" if pd.isna(demo_key) else demo_key,
        "demo_id": "" if pd.isna(demo_id) else demo_id,
        "dim": int(dim_idx),
        "n": int(gt.size),
    }
    if gt.size == 0:
        return row

    diff = pred - gt
    abs_err = np.abs(diff)
    row.update(
        {
            "mae": float(np.mean(abs_err)),
            "mse": float(np.mean(np.square(diff))),
            "rmse": float(np.sqrt(np.mean(np.square(diff)))),
            "median_abs_err": float(np.median(abs_err)),
            "p95_abs_err": float(np.percentile(abs_err, 95)),
            "max_abs_err": float(np.max(abs_err)),
            "mean_signed_err": float(np.mean(diff)),
            "binary_like": False,
        }
    )

    if not _is_binary_like(gt, pred):
        return row

    gt_pos = gt >= 0.5
    pred_pos = pred >= 0.5
    tp = int(np.sum(gt_pos & pred_pos))
    tn = int(np.sum(~gt_pos & ~pred_pos))
    fp = int(np.sum(~gt_pos & pred_pos))
    fn = int(np.sum(gt_pos & ~pred_pos))
    total = tp + tn + fp + fn
    fp_runs = _run_lengths(~gt_pos & pred_pos)
    fn_runs = _run_lengths(gt_pos & ~pred_pos)
    gt_edges = _transition_indices(gt_pos)
    pred_edges = _transition_indices(pred_pos)

    row.update(
        {
            "binary_like": True,
            "tn": tn,
            "fp": fp,
            "fn": fn,
            "tp": tp,
            "accuracy": _safe_ratio(tp + tn, total),
            "precision": _safe_ratio(tp, tp + fp),
            "recall": _safe_ratio(tp, tp + fn),
            "f1": _safe_ratio(2 * tp, 2 * tp + fp + fn),
            "specificity": _safe_ratio(tn, tn + fp),
            "fpr": _safe_ratio(fp, fp + tn),
            "fnr": _safe_ratio(fn, fn + tp),
            "gt_pos_rate": _safe_ratio(tp + fn, total),
            "pred_pos_rate": _safe_ratio(tp + fp, total),
            "fp_burst_count": len(fp_runs),
            "fp_burst_steps": int(sum(fp_runs)),
            "max_fp_burst_len": int(max(fp_runs)) if fp_runs else 0,
            "mean_fp_burst_len": float(np.mean(fp_runs)) if fp_runs else 0.0,
            "fn_burst_count": len(fn_runs),
            "fn_burst_steps": int(sum(fn_runs)),
            "max_fn_burst_len": int(max(fn_runs)) if fn_runs else 0,
            "mean_fn_burst_len": float(np.mean(fn_runs)) if fn_runs else 0.0,
            "gt_transition_count": int(gt_edges.size),
            "pred_transition_count": int(pred_edges.size),
            "extra_transition_count": int(max(pred_edges.size - gt_edges.size, 0)),
            "missed_transition_count": int(max(gt_edges.size - pred_edges.size, 0)),
            "mean_nearest_transition_error": _nearest_transition_error(gt_edges, pred_edges),
        }
    )
    return row


def _read_metric_rows(csv_paths: List[str]) -> Tuple[List[Dict[str, Any]], List[pd.DataFrame]]:
    rows: List[Dict[str, Any]] = []
    frames: List[pd.DataFrame] = []
    for csv_path in csv_paths:
        csv_file = Path(csv_path).expanduser()
        if not csv_file.exists():
            continue
        df = pd.read_csv(csv_file)
        if "checkpoint" not in df.columns:
            df["checkpoint"] = csv_file.stem.replace("results_bc_PatcherBot_", "")
        gt_cols = _sorted_action_columns(df, "gt_")
        pred_cols = _sorted_action_columns(df, "pred_")
        if not gt_cols or len(gt_cols) != len(pred_cols):
            continue

        df["_source_csv"] = csv_file.name
        df["_csv_path"] = str(csv_file)
        frames.append(df)

        group_cols = ["checkpoint"]
        for optional_col in ("demo_key", "demo_id"):
            if optional_col in df.columns:
                group_cols.append(optional_col)
        for group_values, group_df in df.groupby(group_cols, dropna=False):
            if not isinstance(group_values, tuple):
                group_values = (group_values,)
            group_info = dict(zip(group_cols, group_values))
            for dim_idx, (gt_col, pred_col) in enumerate(zip(gt_cols, pred_cols)):
                rows.append(
                    _series_metrics(
                        source_csv=csv_file.name,
                        csv_path=str(csv_file),
                        checkpoint=str(group_info.get("checkpoint", "")),
                        demo_key=group_info.get("demo_key", ""),
                        demo_id=group_info.get("demo_id", ""),
                        dim_idx=dim_idx,
                        gt_values=group_df[gt_col].to_numpy(),
                        pred_values=group_df[pred_col].to_numpy(),
                    )
                )
    return rows, frames


def _checkpoint_metric_rows(frames: List[pd.DataFrame]) -> List[Dict[str, Any]]:
    if not frames:
        return []
    df = pd.concat(frames, ignore_index=True)
    gt_cols = _sorted_action_columns(df, "gt_")
    pred_cols = _sorted_action_columns(df, "pred_")
    rows: List[Dict[str, Any]] = []
    for checkpoint, group_df in df.groupby("checkpoint", dropna=False):
        for dim_idx, (gt_col, pred_col) in enumerate(zip(gt_cols, pred_cols)):
            rows.append(
                _series_metrics(
                    source_csv="",
                    csv_path="",
                    checkpoint=str(checkpoint),
                    demo_key="",
                    demo_id="",
                    dim_idx=dim_idx,
                    gt_values=group_df[gt_col].to_numpy(),
                    pred_values=group_df[pred_col].to_numpy(),
                )
            )
    return rows


def _plot_binary_checkpoint_metrics(metrics_df: pd.DataFrame, out_path: Path) -> Optional[Path]:
    binary_df = metrics_df[metrics_df.get("binary_like", False) == True].copy()
    if binary_df.empty:
        return None

    binary_df = binary_df.sort_values(["checkpoint", "dim"])
    labels = [
        f"{row.checkpoint} d{int(row.dim)}" if len(binary_df["dim"].unique()) > 1 else str(row.checkpoint)
        for row in binary_df.itertuples()
    ]
    x = np.arange(len(binary_df))

    fig_height = max(7.0, 0.28 * len(binary_df) + 5.0)
    fig, axes = plt.subplots(2, 1, figsize=(max(12.0, 0.45 * len(binary_df)), fig_height), sharex=True)

    for metric in ("fpr", "fnr", "precision", "recall", "f1"):
        if metric in binary_df:
            axes[0].plot(x, binary_df[metric].astype(float), marker="o", linewidth=1.6, label=metric)
    axes[0].set_ylabel("Rate")
    axes[0].set_ylim(-0.02, 1.02)
    axes[0].set_title("Binary offline metrics by checkpoint")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc="best", frameon=False)

    width = 0.4
    axes[1].bar(x - width / 2, binary_df["fp_burst_count"].astype(float), width=width, label="FP bursts")
    axes[1].bar(x + width / 2, binary_df["fn_burst_count"].astype(float), width=width, label="FN bursts")
    axes[1].set_ylabel("Burst count")
    axes[1].grid(True, axis="y", alpha=0.3)
    axes[1].legend(loc="best", frameon=False)

    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=65, ha="right")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return out_path


def generate_aggregate_metric_reports(csv_paths: List[str], plot_dir: str) -> Dict[str, Any]:
    plot_root = Path(plot_dir).expanduser()
    plot_root.mkdir(parents=True, exist_ok=True)

    per_csv_rows, frames = _read_metric_rows(csv_paths)
    by_checkpoint_rows = _checkpoint_metric_rows(frames)

    per_csv_path = plot_root / "aggregate_offline_metrics_per_csv.csv"
    by_checkpoint_path = plot_root / "aggregate_offline_metrics_by_checkpoint.csv"
    summary_path = plot_root / "aggregate_offline_metrics_summary.json"
    binary_plot_path = plot_root / "aggregate_binary_metrics_by_checkpoint.png"

    per_csv_df = pd.DataFrame(per_csv_rows)
    by_checkpoint_df = pd.DataFrame(by_checkpoint_rows)
    per_csv_df.to_csv(per_csv_path, index=False)
    by_checkpoint_df.to_csv(by_checkpoint_path, index=False)

    binary_plot_output = None
    if not by_checkpoint_df.empty and "binary_like" in by_checkpoint_df:
        binary_plot_output = _plot_binary_checkpoint_metrics(by_checkpoint_df, binary_plot_path)

    checkpoint_count = int(by_checkpoint_df["checkpoint"].nunique()) if "checkpoint" in by_checkpoint_df else 0
    binary_metrics_emitted = bool(
        not by_checkpoint_df.empty
        and "binary_like" in by_checkpoint_df
        and by_checkpoint_df["binary_like"].fillna(False).astype(bool).any()
    )
    summary = {
        "generated_at_utc": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "source_csv_count": int(len([p for p in csv_paths if Path(p).expanduser().exists()])),
        "metric_source_csv_count": int(len(frames)),
        "checkpoint_count": checkpoint_count,
        "per_csv_metric_rows": int(len(per_csv_df)),
        "by_checkpoint_metric_rows": int(len(by_checkpoint_df)),
        "binary_metrics_emitted": binary_metrics_emitted,
        "output_files": {
            "per_csv": str(per_csv_path),
            "by_checkpoint": str(by_checkpoint_path),
            "summary": str(summary_path),
            "binary_plot": str(binary_plot_output) if binary_plot_output is not None else None,
        },
    }
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)

    print(f"Saved aggregate metrics: {per_csv_path}")
    print(f"Saved aggregate checkpoint metrics: {by_checkpoint_path}")
    print(f"Saved aggregate summary: {summary_path}")
    if binary_plot_output is not None:
        print(f"Saved aggregate binary chart: {binary_plot_output}")
    return summary


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
    successful_csvs = [result["csv_path"] for result in results if "error" not in result and result.get("csv_path")]
    if successful_csvs:
        try:
            generate_aggregate_metric_reports(successful_csvs, str(plot_root))
        except Exception as exc:
            print(f"[ERROR] Failed to generate aggregate metrics: {exc}")
    return results


__all__ = [
    "FFT_BAND_RATIOS",
    "batch_generate_plots",
    "default_plot_paths_for_csv",
    "generate_aggregate_metric_reports",
    "generate_prediction_plots",
]
