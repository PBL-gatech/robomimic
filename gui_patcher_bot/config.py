#!/usr/bin/env python3
"""Configuration management for the PatcherBot evaluation GUI."""

from pathlib import Path


def _sanitize_name(value: str) -> str:
    return Path(value).name.replace("/", "_").replace("\\", "_")


class Config:
    """Central configuration for the evaluation GUI."""

    DEFAULT_CHECKPOINT_DIR = r"C:\Users\sa-forest\Documents\GitHub\robomimic\bc_patcherBot\Gigasealing"
    DEFAULT_DATASET_PATH = r"C:\Users\sa-forest\Documents\GitHub\PatcherBot-Agent\experiments\Datasets\PatcherBot_test_dataset_v0_840\PatcherBot_test_dataset_v0_840_gigaseal.hdf5"
    DEFAULT_CSV_DIR = r"C:\Users\sa-forest\Documents\GitHub\robomimic\bc_patcherBot\Gigasealing\results\v0_840"
    DEFAULT_METADATA_DIR = DEFAULT_CSV_DIR
    DEFAULT_PLOTS_DIR = r"C:\Users\sa-forest\Documents\GitHub\robomimic\bc_patcherBot\Gigasealing\results\v0_840\plots"

    def __init__(self):
        self.checkpoint_dir = self.DEFAULT_CHECKPOINT_DIR
        self.dataset_path = self.DEFAULT_DATASET_PATH
        self.csv_dir = self.DEFAULT_CSV_DIR
        self.metadata_dir = self.DEFAULT_METADATA_DIR
        self.plots_dir = self.DEFAULT_PLOTS_DIR
        self.output_dir = self.DEFAULT_CSV_DIR
        self.horizon = None
        self.eps = -1.0
        self.show_pos_traj = True
        self.frame_stack = None
        self.demo_id = None
        self.evaluate_all_demos = True

    def update(self, **kwargs):
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
        self.sync_output_dirs()

    def sync_output_dirs(self):
        csv_dir = self.resolve_csv_dir()
        if not self.metadata_dir:
            self.metadata_dir = str(csv_dir)
        if not self.plots_dir:
            self.plots_dir = str(csv_dir / "plots")
        self.output_dir = str(csv_dir)

    def resolve_csv_dir(self) -> Path:
        return Path(self.csv_dir or self.DEFAULT_CSV_DIR).expanduser()

    def resolve_metadata_dir(self) -> Path:
        return Path(self.metadata_dir or self.resolve_csv_dir()).expanduser()

    def resolve_plots_dir(self) -> Path:
        fallback = self.resolve_csv_dir() / "plots"
        return Path(self.plots_dir or fallback).expanduser()

    def get_checkpoint_paths(self):
        path = Path(self.checkpoint_dir).expanduser()
        if not path.exists():
            return []
        return sorted(p for p in path.glob("*.pth") if p.is_file())

    def get_csv_path(self, checkpoint_path: str, demo_key: str = None) -> Path:
        csv_dir = self.resolve_csv_dir()
        stem = f"results_bc_PatcherBot_{_sanitize_name(checkpoint_path)}"
        if demo_key is not None:
            stem = f"{stem}_{_sanitize_name(demo_key)}"
        return csv_dir / f"{stem}.csv"

    def get_metadata_path(self, checkpoint_path: str, demo_key: str = None) -> Path:
        metadata_dir = self.resolve_metadata_dir()
        stem = f"metadata_bc_PatcherBot_{_sanitize_name(checkpoint_path)}"
        if demo_key is not None:
            stem = f"{stem}_{_sanitize_name(demo_key)}"
        return metadata_dir / f"{stem}.json"

    def get_plot_path(self, csv_path: str) -> Path:
        plots_dir = self.resolve_plots_dir()
        ckpt_stem = Path(csv_path).stem.replace("results_bc_PatcherBot_", "")
        return plots_dir / f"predictions_{ckpt_stem}.png"

    def get_fft_path(self, csv_path: str) -> Path:
        plots_dir = self.resolve_plots_dir()
        ckpt_stem = Path(csv_path).stem.replace("results_bc_PatcherBot_", "")
        return plots_dir / f"fft_{ckpt_stem}.png"


def load_config(config_file=None):
    return Config()
