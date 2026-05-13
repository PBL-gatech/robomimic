#!/usr/bin/env python3
"""Thin GUI wrapper around robomimic's shared PatcherBot plotting utilities."""

from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from robomimic.utils.patcherbot_visualization import (  # noqa: E402
    FFT_BAND_RATIOS,
    batch_generate_plots,
    default_plot_paths_for_csv,
    generate_aggregate_metric_reports,
    generate_prediction_plots,
)


__all__ = [
    "FFT_BAND_RATIOS",
    "batch_generate_plots",
    "default_plot_paths_for_csv",
    "generate_aggregate_metric_reports",
    "generate_prediction_plots",
]
