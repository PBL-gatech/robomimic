#!/usr/bin/env python3
"""Thin GUI wrapper over robomimic's shared PatcherBot evaluation utilities."""

from pathlib import Path
from typing import Any, Dict, List

try:
    from .config import Config
    from .robomimic_compat import evaluate_patcherbot_checkpoint, evaluate_patcherbot_checkpoints
    from .visualization import (
        default_plot_paths_for_csv,
        generate_aggregate_metric_reports,
        generate_prediction_plots,
    )
except ImportError:
    from config import Config
    from robomimic_compat import evaluate_patcherbot_checkpoint, evaluate_patcherbot_checkpoints
    from visualization import (
        default_plot_paths_for_csv,
        generate_aggregate_metric_reports,
        generate_prediction_plots,
    )


class EvaluationManager:
    """Bridge between the GUI config and robomimic's evaluation helpers."""

    def __init__(self, config: Config):
        self.config = config

    def _sync_config(self):
        self.config.sync_output_dirs()

    def evaluate_checkpoint(self, checkpoint_path: str) -> Dict[str, Any]:
        self._sync_config()
        return evaluate_patcherbot_checkpoint(checkpoint_path, self.config)

    def evaluate_all_checkpoints(self, checkpoint_dir: str = None) -> List[Dict[str, Any]]:
        if checkpoint_dir is not None:
            self.config.checkpoint_dir = checkpoint_dir
        self._sync_config()
        return evaluate_patcherbot_checkpoints(self.config.checkpoint_dir, self.config)

    def generate_plot_bundle(self, csv_path: str) -> Dict[str, str]:
        self._sync_config()
        out_png, out_fft = default_plot_paths_for_csv(csv_path, str(self.config.resolve_plots_dir()))
        plot_result = generate_prediction_plots(
            csv_path=csv_path,
            out_png_path=str(out_png),
            out_fft_png_path=str(out_fft),
        )
        return {
            "main": plot_result["output_png"],
            "fft": plot_result.get("fft_png"),
        }

    def enrich_result_with_plots(self, result: Dict[str, Any]) -> Dict[str, Any]:
        csv_path = Path(result.get("csv_path", ""))
        if result.get("success") and csv_path.exists():
            result["plot_paths"] = self.generate_plot_bundle(str(csv_path))
        return result

    def generate_aggregate_metric_reports(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        self._sync_config()
        csv_paths = []
        for result in results:
            csv_path = Path(result.get("csv_path", ""))
            if result.get("success") and csv_path.exists():
                csv_paths.append(str(csv_path))
        return generate_aggregate_metric_reports(csv_paths, str(self.config.resolve_plots_dir()))
