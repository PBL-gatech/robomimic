#!/usr/bin/env python3
"""Compatibility layer that forwards GUI imports into robomimic utilities."""

from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from robomimic.utils.patcherbot_eval import (  # noqa: E402
    _append_vector,
    _build_obs_specs,
    _build_step_record,
    _compute_loss_components,
    _ensure_key_in_modalities,
    _extract_obs_modalities,
    _extract_policy_action,
    _infer_derived_spec,
    _infer_low_dim_shape,
    _infer_rgb_shape,
    _json_safe,
    _resolve_checkpoints,
    _safe_get,
    _summarize_losses,
    _to_list,
    evaluate_patcherbot_checkpoint,
    evaluate_patcherbot_checkpoints,
    get_checkpoint_csv_path,
    get_checkpoint_metadata_path,
    sanitize_checkpoint_name,
)


__all__ = [
    "_append_vector",
    "_build_obs_specs",
    "_build_step_record",
    "_compute_loss_components",
    "_ensure_key_in_modalities",
    "_extract_obs_modalities",
    "_extract_policy_action",
    "_infer_derived_spec",
    "_infer_low_dim_shape",
    "_infer_rgb_shape",
    "_json_safe",
    "_resolve_checkpoints",
    "_safe_get",
    "_summarize_losses",
    "_to_list",
    "evaluate_patcherbot_checkpoint",
    "evaluate_patcherbot_checkpoints",
    "get_checkpoint_csv_path",
    "get_checkpoint_metadata_path",
    "sanitize_checkpoint_name",
]
