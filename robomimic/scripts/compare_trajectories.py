#!/usr/bin/env python3
"""Compare predicted trajectories across multiple CSV files.

This script loads evaluation CSV files (such as the ones produced by
`scripts/run_PatcherBot_agent.py`) from a folder, groups trajectories by
checkpoint, and visualizes:
  * predictions from every checkpoint in each file
  * the corresponding ground-truth traces
  * the absolute and normalized errors for every checkpoint trajectory

All trajectories are clamped to the shortest sequence length observed over all
files and checkpoints so that curves align cleanly when plotted. The figure
layout matches :mod:`visualize_results`, with predictions on the top row,
absolute error in the middle, and normalized error on the bottom.

Legend rules:
  * ground-truth traces are always black (distinct dashed patterns per file)
  * predictions for a file share a unique color across all checkpoints

Example usage::

    python -m robomimic.scripts.compare_trajectories \
        --csv-dir bc_patcherBot/PipetteFinding/results \
        --pattern "results_bc_PatcherBot_v0_010_*.csv" \
        --out "comparison_bc_PatcherBot_v0_010.png"

"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from itertools import cycle
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


@dataclass
class CheckpointRecord:
    file_label: str
    checkpoint: str
    steps: np.ndarray
    gt: Dict[str, np.ndarray]
    pred: Dict[str, np.ndarray]
    abs_error: Dict[str, np.ndarray]
    norm_error: Dict[str, np.ndarray]


@dataclass
class FileSeries:
    label: str
    steps: np.ndarray
    gt: Dict[str, np.ndarray]


@dataclass
class Normalization:
    mode: str
    denominators: Mapping[str, float]

    def describe(self, dim: str) -> str:
        denom = self.denominators.get(dim, float("nan"))
        return f"mode={self.mode}, denom={denom:.6f}"


_SUPPORTED_NORMALIZATIONS = {"gt_range", "gt_std", "abs_max", "global_max", "none"}


def _sorted_dim_keys(keys: Iterable[str]) -> List[str]:
    def sort_key(name: str) -> Sequence[int | str]:
        try:
            return (0, int(name))
        except ValueError:
            return (1, name)

    return sorted(keys, key=sort_key)


def _discover_dimensions(df: pd.DataFrame) -> List[str]:
    gt_cols = [col for col in df.columns if col.startswith("gt_")]
    dims = []
    for gt_col in gt_cols:
        suffix = gt_col[len("gt_") :]
        if f"pred_{suffix}" in df.columns:
            dims.append(suffix)
    if not dims:
        raise ValueError("CSV must contain matching 'gt_<dim>' and 'pred_<dim>' columns")
    return _sorted_dim_keys(dims)


def _split_by_checkpoint(df: pd.DataFrame) -> List[Tuple[str, pd.DataFrame]]:
    if "checkpoint" not in df.columns:
        return [("run_0", df.reset_index(drop=True))]

    checkpoint_str = df["checkpoint"].astype(str)
    ordered = list(dict.fromkeys(checkpoint_str))
    groups: List[Tuple[str, pd.DataFrame]] = []
    for ckpt in ordered:
        ckpt_mask = checkpoint_str == ckpt
        ckpt_df = df.loc[ckpt_mask]
        ckpt_df = ckpt_df.sort_values("step", kind="mergesort").reset_index(drop=True)
        groups.append((ckpt, ckpt_df))
    return groups


def _load_csvs(csv_paths: Sequence[Path]) -> Dict[Path, pd.DataFrame]:
    loaded: Dict[Path, pd.DataFrame] = {}
    for path in csv_paths:
        df = pd.read_csv(path)
        if df.empty:
            raise ValueError(f"CSV '{path}' is empty")
        if "step" in df.columns:
            sort_cols: Sequence[str] | str
            if "checkpoint" in df.columns:
                sort_cols = ["checkpoint", "step"]
            else:
                sort_cols = "step"
            df = df.sort_values(sort_cols, kind="mergesort").reset_index(drop=True)
        else:
            df.insert(0, "step", np.arange(len(df), dtype=float))
        loaded[path] = df
    return loaded


def _collect_records(
    dataframes: Mapping[Path, pd.DataFrame],
    dims: Sequence[str],
) -> Tuple[List[CheckpointRecord], Dict[str, FileSeries], int]:
    per_file_groups: Dict[str, List[Tuple[str, pd.DataFrame]]] = {}
    min_length: Optional[int] = None

    for path, df in dataframes.items():
        label = path.stem
        groups = _split_by_checkpoint(df)
        if not groups:
            raise ValueError(f"No checkpoint groups found in '{path}'")
        per_file_groups[label] = groups
        for _, gdf in groups:
            length = len(gdf)
            if min_length is None or length < min_length:
                min_length = length

    if min_length is None or min_length <= 0:
        raise ValueError("Unable to determine a positive sequence length across checkpoints")

    records: List[CheckpointRecord] = []
    file_series: Dict[str, FileSeries] = {}
    truncated_any = False

    for label, groups in per_file_groups.items():
        for idx, (ckpt, gdf) in enumerate(groups):
            if len(gdf) > min_length:
                truncated_any = True
            truncated = gdf.iloc[:min_length].reset_index(drop=True)
            steps = truncated["step"].to_numpy(copy=False)
            gt: Dict[str, np.ndarray] = {}
            pred: Dict[str, np.ndarray] = {}
            abs_err: Dict[str, np.ndarray] = {}
            norm_err: Dict[str, np.ndarray] = {}
            for dim in dims:
                gt_col, pred_col = f"gt_{dim}", f"pred_{dim}"
                gt_values = truncated[gt_col].to_numpy(copy=False)
                pred_values = truncated[pred_col].to_numpy(copy=False)
                err = pred_values - gt_values
                gt[dim] = gt_values
                pred[dim] = pred_values
                abs_err[dim] = np.abs(err)
                norm_err[dim] = np.zeros_like(abs_err[dim])
            records.append(
                CheckpointRecord(
                    file_label=label,
                    checkpoint=str(ckpt),
                    steps=steps,
                    gt=gt,
                    pred=pred,
                    abs_error=abs_err,
                    norm_error=norm_err,
                )
            )
            if idx == 0:
                file_series[label] = FileSeries(label=label, steps=steps, gt={dim: gt[dim].copy() for dim in dims})

    if truncated_any:
        print(f"[INFO] Clamped all trajectories to the first {min_length} steps across files/checkpoints")

    return records, file_series, min_length


def _compile_denominators(
    file_series: Mapping[str, FileSeries],
    records: Sequence[CheckpointRecord],
    dims: Sequence[str],
    mode: str,
) -> Normalization:
    denominators: Dict[str, float] = {}

    if mode == "global_max":
        global_max = 0.0
        for dim in dims:
            dim_max = 0.0
            for rec in records:
                dim_max = max(dim_max, float(rec.abs_error[dim].max()))
            global_max = max(global_max, dim_max)
        denom = global_max if np.isfinite(global_max) and global_max > 0 else 1.0
        for dim in dims:
            denominators[dim] = denom
        return Normalization(mode=mode, denominators=denominators)

    for dim in dims:
        gt_arrays = [series.gt[dim] for series in file_series.values()]
        joined = np.concatenate(gt_arrays)
        if not np.isfinite(joined).all():
            raise ValueError(f"Non-finite values detected in ground truth for dimension '{dim}'")
        if mode == "gt_range":
            denom = float(joined.max() - joined.min())
        elif mode == "gt_std":
            denom = float(joined.std(ddof=0))
        elif mode == "abs_max":
            denom = float(np.abs(joined).max())
        elif mode == "none":
            denom = 1.0
        else:
            raise ValueError(f"Unsupported normalization mode: {mode}")
        if not np.isfinite(denom) or denom <= 0.0:
            denom = 1.0
        denominators[dim] = denom
    return Normalization(mode=mode, denominators=denominators)


def _apply_normalization(records: Sequence[CheckpointRecord], normalization: Normalization) -> None:
    for rec in records:
        for dim, abs_err in rec.abs_error.items():
            denom = normalization.denominators.get(dim, 1.0)
            rec.norm_error[dim] = abs_err / denom


def _build_figure(
    records: Sequence[CheckpointRecord],
    file_series: Mapping[str, FileSeries],
    dims: Sequence[str],
    normalization: Normalization,
    title: str,
    dpi: int,
) -> plt.Figure:
    n_dims = len(dims)
    if n_dims == 0:
        raise ValueError("No action dimensions discovered.")

    fig_width = max(12.0, 4.5 * n_dims)
    fig_height = 10.5
    fig = plt.figure(figsize=(fig_width, fig_height))
    grid = plt.GridSpec(3, n_dims, figure=fig, height_ratios=[2.5, 1.0, 1.0], hspace=0.4, wspace=0.35)

    pred_axes = [fig.add_subplot(grid[0, dim_idx]) for dim_idx in range(n_dims)]
    raw_axes = [fig.add_subplot(grid[1, dim_idx]) for dim_idx in range(n_dims)]
    norm_axes = [fig.add_subplot(grid[2, dim_idx]) for dim_idx in range(n_dims)]

    file_labels = list(file_series.keys())
    cmap = plt.get_cmap("tab10", max(len(file_labels), 1))
    color_map = {label: cmap(idx % cmap.N) for idx, label in enumerate(file_labels)}
    gt_style_cycle = cycle(["--", "-.", ":", (0, (2, 1)), (0, (3, 1, 1, 1))])
    gt_style_map = {label: next(gt_style_cycle) for label in file_labels}

    norm_label = "|gt - pred|" if normalization.mode == "none" else "Normalized |gt - pred|"

    pred_labels_drawn: Dict[str, bool] = {label: False for label in file_labels}
    gt_labels_drawn: Dict[str, bool] = {label: False for label in file_labels}

    for dim_idx, dim in enumerate(dims):
        pred_ax = pred_axes[dim_idx]
        err_ax = raw_axes[dim_idx]
        norm_ax = norm_axes[dim_idx]

        # Plot predictions for every checkpoint, grouping color by source file.
        for rec in records:
            color = color_map[rec.file_label]
            pred_label = f"{rec.file_label} pred" if not pred_labels_drawn[rec.file_label] and dim_idx == 0 else None
            if pred_label:
                pred_labels_drawn[rec.file_label] = True
            pred_ax.plot(
                rec.steps,
                rec.pred[dim],
                label=pred_label,
                color=color,
                linewidth=1.5,
                alpha=0.85,
            )
            err_ax.plot(
                rec.steps,
                rec.abs_error[dim],
                label=None,
                color=color,
                linewidth=1.4,
                alpha=0.75,
            )
            norm_ax.plot(
                rec.steps,
                rec.norm_error[dim],
                label=None,
                color=color,
                linewidth=1.4,
                alpha=0.75,
            )

        # Overlay ground-truth traces (one per file, always black with unique styles).
        for label in file_labels:
            series = file_series[label]
            gt_label = f"{label} gt" if not gt_labels_drawn[label] and dim_idx == 0 else None
            if gt_label:
                gt_labels_drawn[label] = True
            pred_ax.plot(
                series.steps,
                series.gt[dim],
                label=gt_label,
                color="black",
                linestyle=gt_style_map[label],
                linewidth=2.0,
                alpha=0.9,
            )

        pred_ax.set_title(f"Dim {dim}: prediction vs GT")
        pred_ax.set_ylabel("Value")
        pred_ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.4)

        err_ax.set_title(f"Dim {dim}: absolute error")
        err_ax.set_ylabel("|gt - pred|")
        err_ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.4)

        if normalization.mode != "none":
            norm_ax.set_title(f"Dim {dim}: normalized error ({normalization.describe(dim)})")
        else:
            norm_ax.set_title(f"Dim {dim}: |gt - pred|")
        norm_ax.set_xlabel("Step")
        norm_ax.set_ylabel(norm_label)
        norm_ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.4)

    legend_handles, legend_labels = pred_axes[0].get_legend_handles_labels()
    if legend_handles:
        fig.legend(
            legend_handles,
            legend_labels,
            loc="center left",
            bbox_to_anchor=(0.99, 0.5),
            frameon=False,
        )

    fig.suptitle(title, fontsize=14)
    fig.tight_layout(rect=[0, 0, 0.85, 0.94])
    fig.set_dpi(dpi)
    return fig


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv-dir", default=r"C:\Users\sa-forest\Documents\GitHub\robomimic\robomimic\scripts\debug\v10", type=Path,  help="Folder containing CSV files to compare")
    parser.add_argument(
        "--pattern",
        default="*.csv",
        help="Glob pattern (relative to csv-dir) used to select CSV files",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional maximum number of CSV files to include (after sorting)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Path to save the output PNG. Defaults to <csv-dir>/trajectory_comparison.png",
    )
    parser.add_argument(
        "--normalization",
        choices=sorted(_SUPPORTED_NORMALIZATIONS),
        default="global_max",
        help="Statistic used to normalize the absolute error",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=140,
        help="Figure DPI when saving",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display the plot interactively in addition to saving",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print per-file summary statistics",
    )
    return parser.parse_args(argv)


def _summarize(records: Sequence[CheckpointRecord], dims: Sequence[str]) -> None:
    for rec in records:
        print(f"[FILE] {rec.file_label} | checkpoint={rec.checkpoint}")
        for dim in dims:
            abs_err = rec.abs_error[dim]
            mean_abs = float(abs_err.mean())
            max_abs = float(abs_err.max())
            print(
                f"  - Dim {dim}: mean|err|={mean_abs:.6f}, max|err|={max_abs:.6f}, steps={len(abs_err)}"
            )


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = _parse_args(argv)

    csv_dir = args.csv_dir.expanduser().resolve()
    if not csv_dir.is_dir():
        sys.exit(f"CSV directory not found: {csv_dir}")

    csv_paths = sorted(csv_dir.glob(args.pattern))
    if args.limit is not None:
        csv_paths = csv_paths[: args.limit]

    if not csv_paths:
        sys.exit("No CSV files matched the provided directory/pattern")

    print(f"[INFO] Loaded {len(csv_paths)} CSV files from {csv_dir}")
    for path in csv_paths:
        print(f" - {path.name}")

    dataframes = _load_csvs(csv_paths)
    dims = _discover_dimensions(next(iter(dataframes.values())))

    records, file_series, min_length = _collect_records(dataframes, dims)
    print(f"[INFO] Prepared {len(records)} checkpoint trajectories at {min_length} steps each")

    normalization = _compile_denominators(file_series, records, dims, args.normalization)
    _apply_normalization(records, normalization)

    if args.verbose:
        _summarize(records, dims)

    out_path = args.out
    if out_path is None:
        out_path = csv_dir / "trajectory_comparison.png"
    else:
        out_path = args.out.expanduser().resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)

    title = f"Trajectory comparison ({', '.join(dim for dim in dims)})"
    fig = _build_figure(records, file_series, dims, normalization, title=title, dpi=args.dpi)
    fig.savefig(out_path, dpi=args.dpi)
    print(f"[RESULT] Saved comparison figure to {out_path}")

    if args.show:
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    main()
