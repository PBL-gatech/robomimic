"""Lightweight helpers for exporting Robomimic policies to ONNX."""
from __future__ import annotations

import argparse
import glob
import json
import pathlib
import re
from collections import OrderedDict
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn

from robomimic.utils.file_utils import policy_from_checkpoint
from robomimic.utils import obs_utils as ObsUtils
import robomimic.utils.tensor_utils as TensorUtils


# ---------------------------------------------------------------------------
# CLI helper
# ---------------------------------------------------------------------------

def make_export_arg_parser(default_folder: Optional[str] = None) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export a Robomimic policy to ONNX")
    parser.add_argument("--ckpt", help="Path to .pth checkpoint")
    parser.add_argument("--config", help="Path to config.json")
    parser.add_argument("--out", help="Output ONNX file")
    parser.add_argument("--folder", default=default_folder, help="Directory containing config and checkpoint")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"], help="Device to load the checkpoint on")
    return parser


# ---------------------------------------------------------------------------
# Filesystem helpers
# ---------------------------------------------------------------------------

def find_paths(folder: str) -> Tuple[pathlib.Path, pathlib.Path, pathlib.Path]:
    folder_path = pathlib.Path(folder).expanduser()
    configs = list(folder_path.rglob("config.json"))
    if not configs:
        raise FileNotFoundError(f"Config file not found under {folder_path}")
    config_path = configs[0]

    ckpts = glob.glob(str(folder_path / "**" / "model_epoch_*.pth"), recursive=True)
    if not ckpts:
        raise FileNotFoundError(f"Checkpoint not found under {folder_path}")
    ckpts = sorted(ckpts, key=lambda path: int(re.search(r"model_epoch_(\d+)", path).group(1)))
    ckpt_path = pathlib.Path(ckpts[-1])

    onnx_path = folder_path / f"{ckpt_path.stem}.onnx"
    return config_path, ckpt_path, onnx_path


def load_config(config_path: pathlib.Path) -> Dict[str, Any]:
    with open(pathlib.Path(config_path).expanduser(), "r") as f:
        return json.load(f)


def load_policy(ckpt_path: pathlib.Path, device: str) -> Tuple[Any, Dict[str, Any]]:
    policy, ckpt_dict = policy_from_checkpoint(
        ckpt_path=pathlib.Path(ckpt_path).expanduser(),
        device=torch.device(device),
        verbose=False,
    )
    return policy, ckpt_dict


# ---------------------------------------------------------------------------
# Normalization extraction
# ---------------------------------------------------------------------------

def _to_1d_np(x: Any) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    return np.asarray(x, dtype=np.float32).reshape(-1)


def extract_action_normalization(ckpt_dict: Mapping[str, Any]) -> Optional[Dict[str, Dict[str, torch.Tensor]]]:
    def _find_stats(node: Any) -> Optional[Mapping[str, Any]]:
        if not isinstance(node, Mapping):
            return None
        if ("offset" in node and "scale" in node) or ("mean" in node and "std" in node):
            return node
        for value in node.values():
            found = _find_stats(value)
            if found is not None:
                return found
        return None

    stats: Optional[Mapping[str, Any]] = None
    if isinstance(ckpt_dict, Mapping):
        candidates = [
            ("normalization_stats", "actions"),
            ("action_normalization_stats", "actions"),
            ("action_normalization_stats", None),
            ("stats", "actions"),
            ("normalization", "actions"),
        ]
        for top, sub in candidates:
            node = ckpt_dict.get(top)
            if node is None:
                continue
            stats = _find_stats(node if sub is None else node.get(sub))
            if stats is not None:
                break
        if stats is None:
            stats = _find_stats(ckpt_dict)

    if stats is None:
        return None

    if "offset" in stats and "scale" in stats:
        offset, scale = stats["offset"], stats["scale"]
    elif "mean" in stats and "std" in stats:
        offset, scale = stats["mean"], stats["std"]
    else:
        return None

    offset_np = _to_1d_np(offset)
    scale_np = _to_1d_np(scale)

    offset_t = torch.from_numpy(offset_np).view(1, offset_np.shape[0])
    scale_t = torch.from_numpy(scale_np).view(1, scale_np.shape[0])
    return {"actions": {"offset": offset_t, "scale": scale_t}}


def extract_obs_normalization(policy: Any, obs_keys: Sequence[str], goal_keys: Sequence[str]) -> OrderedDict:
    stats = getattr(policy, "obs_normalization_stats", None)
    if stats is None:
        return OrderedDict()

    ordered: List[str] = []
    for key in list(obs_keys) + list(goal_keys):
        if key not in ordered:
            ordered.append(key)

    filtered = OrderedDict((key, stats[key]) for key in ordered if key in stats)
    if not filtered:
        return OrderedDict()

    return TensorUtils.to_float(TensorUtils.to_tensor(filtered))


def save_normalization_npz(
    obs_norm: Mapping[str, Mapping[str, torch.Tensor]],
    action_norm: Optional[Mapping[str, Mapping[str, torch.Tensor]]],
    npz_path: pathlib.Path,
) -> None:
    arrays: Dict[str, np.ndarray] = {}
    for key, stat in (obs_norm or {}).items():
        arrays[f"obs::{key}::offset"] = stat["offset"].detach().cpu().numpy()
        arrays[f"obs::{key}::scale"] = stat["scale"].detach().cpu().numpy()
    if action_norm and "actions" in action_norm:
        arrays["actions::offset"] = action_norm["actions"]["offset"].detach().cpu().numpy()
        arrays["actions::scale"] = action_norm["actions"]["scale"].detach().cpu().numpy()
    np.savez(pathlib.Path(npz_path).expanduser(), **arrays)




def save_observation_normalization_npz(
    obs_norm: Mapping[str, Mapping[str, torch.Tensor]],
    npz_path: pathlib.Path,
) -> bool:
    """Save per-observation normalization statistics to a standalone NPZ file."""
    if not obs_norm:
        return False
    arrays: Dict[str, np.ndarray] = {}
    for key, stat in obs_norm.items():
        if not isinstance(stat, Mapping):
            continue
        offset = stat.get("offset")
        scale = stat.get("scale")
        if offset is None or scale is None:
            continue
        arrays[f"{key}::offset"] = offset.detach().cpu().numpy()
        arrays[f"{key}::scale"] = scale.detach().cpu().numpy()
    if not arrays:
        return False
    np.savez(pathlib.Path(npz_path).expanduser(), **arrays)
    return True


def save_action_normalization_npz(
    action_norm: Optional[Mapping[str, Mapping[str, torch.Tensor]]],
    npz_path: pathlib.Path,
) -> bool:
    """Save action normalization statistics to a standalone NPZ file."""
    if not action_norm:
        return False
    stats = action_norm.get("actions") if isinstance(action_norm, Mapping) else None
    if not isinstance(stats, Mapping):
        return False
    offset = stats.get("offset")
    scale = stats.get("scale")
    if offset is None or scale is None:
        return False
    arrays = {
        "offset": offset.detach().cpu().numpy(),
        "scale": scale.detach().cpu().numpy(),
    }
    np.savez(pathlib.Path(npz_path).expanduser(), **arrays)
    return True


# ---------------------------------------------------------------------------
# Actor wrapper and dummy inputs
# ---------------------------------------------------------------------------

def _resolve_actor(policy_obj: Any) -> Any:
    inner = policy_obj.policy
    actor_net = inner
    if hasattr(inner, "nets"):
        nets = inner.nets
        if isinstance(nets, (dict, torch.nn.ModuleDict)):
            if "policy" in nets:
                actor_net = nets["policy"]
            elif "actor" in nets:
                actor_net = nets["actor"]
            else:
                actor_net = next(iter(nets.values()))
    return actor_net


def build_actor_export_module(
    policy_obj: Any,
    config: Mapping[str, Any],
) -> Tuple[torch.nn.Module, List[str], List[str], bool, Mapping[str, Sequence[int]], Mapping[str, Sequence[int]], Mapping[str, Any]]:
    actor_net = _resolve_actor(policy_obj)
    if hasattr(policy_obj.policy, "set_eval"):
        policy_obj.policy.set_eval()
    if hasattr(actor_net, "eval"):
        actor_net = actor_net.eval()

    if hasattr(actor_net, "obs_shapes"):
        obs_shapes = OrderedDict((k, tuple(v)) for k, v in actor_net.obs_shapes.items())
    elif hasattr(policy_obj.policy, "obs_shapes"):
        obs_shapes = OrderedDict((k, tuple(v)) for k, v in policy_obj.policy.obs_shapes.items())
    else:
        raise AttributeError("Actor network missing obs_shapes")

    if hasattr(actor_net, "goal_shapes") and actor_net.goal_shapes:
        goal_shapes = OrderedDict((k, tuple(v)) for k, v in actor_net.goal_shapes.items())
    elif hasattr(policy_obj.policy, "goal_shapes") and policy_obj.policy.goal_shapes:
        goal_shapes = OrderedDict((k, tuple(v)) for k, v in policy_obj.policy.goal_shapes.items())
    else:
        goal_shapes = OrderedDict()

    algo_cfg = config.get("algo", {}) if isinstance(config, Mapping) else {}
    rnn_cfg = algo_cfg.get("rnn", {}) if isinstance(algo_cfg, Mapping) else {}
    is_recurrent = bool(rnn_cfg.get("enabled", False))

    obs_keys = list(obs_shapes.keys())
    goal_keys = list(goal_shapes.keys())

    class OnnxPolicy(nn.Module):
        def __init__(self):
            super().__init__()
            self.actor_net = actor_net.to(torch.device("cpu"))
            self.obs_keys = obs_keys
            self.goal_keys = goal_keys
            self.is_goal_conditioned = len(goal_keys) > 0
            self.is_recurrent = is_recurrent
            self.obs_shapes = obs_shapes
            self.goal_shapes = goal_shapes
            self.output_names = ["actions"] + (["h1", "c1"] if self.is_recurrent else [])
            self.obs_norm_stats = None
            self.action_norm_stats = None

        def set_normalization_stats(self, obs_stats=None, action_stats=None):
            device = torch.device("cpu")
            if obs_stats:
                stats = TensorUtils.to_tensor(obs_stats)
                stats = TensorUtils.to_float(stats)
                stats = TensorUtils.to_device(stats, device)
                self.obs_norm_stats = stats
            else:
                self.obs_norm_stats = None
            if action_stats:
                stats = TensorUtils.to_tensor(action_stats)
                stats = TensorUtils.to_float(stats)
                stats = TensorUtils.to_device(stats, device)
                self.action_norm_stats = stats
            else:
                self.action_norm_stats = None
            return self

        def _build_dict(self, keys: Sequence[str], tensors: Sequence[torch.Tensor]) -> OrderedDict:
            data = OrderedDict()
            for key, tensor in zip(keys, tensors):
                data[key] = tensor.to(torch.float32)
            return data

        def _process_obs(self, tensors: OrderedDict) -> OrderedDict:
            if not tensors:
                return tensors
            processed = ObsUtils.process_obs_dict(tensors)
            if self.obs_norm_stats:
                present = [k for k in tensors.keys() if k in self.obs_norm_stats]
                if present:
                    stats = {k: self.obs_norm_stats[k] for k in present}
                    processed = ObsUtils.normalize_dict(processed, normalization_stats=stats)
            return OrderedDict((k, processed[k]) for k in tensors.keys())

        def _process_goal(self, tensors: OrderedDict) -> OrderedDict:
            if not tensors:
                return tensors
            processed = self._process_obs(tensors)
            sanitized = OrderedDict()
            for key, tensor in processed.items():
                target_shape = self.goal_shapes.get(key)
                if target_shape is None:
                    sanitized[key] = tensor
                    continue
                target_ndim = len(target_shape) + 1
                while tensor.dim() > target_ndim and tensor.shape[1] == 1:
                    tensor = tensor.squeeze(1)
                sanitized[key] = tensor
            return sanitized

        def _postprocess_actions(self, actions):
            if self.action_norm_stats is None:
                return actions
            if isinstance(actions, torch.Tensor):
                stats = self.action_norm_stats.get("actions")
                if stats is None:
                    return actions
                offset = stats["offset"].to(actions.device)
                scale = stats["scale"].to(actions.device)
                result = actions * scale + offset
                return result
            if isinstance(actions, dict):
                stats = {k: self.action_norm_stats[k] for k in actions if k in self.action_norm_stats}
                if not stats:
                    return actions
                stats = TensorUtils.to_device(stats, next(iter(actions.values())).device)
                return ObsUtils.unnormalize_dict(actions, normalization_stats=stats)
            if isinstance(actions, (list, tuple)) and actions:
                first = self._postprocess_actions(actions[0])
                remainder = actions[1:]
                if isinstance(actions, list):
                    return [first, *remainder]
                return (first, *remainder)
            return actions

        def forward(self, *inputs: torch.Tensor):
            idx = 0
            obs_count = len(self.obs_keys)
            goal_count = len(self.goal_keys)

            obs_inputs = inputs[idx: idx + obs_count]
            idx += obs_count
            obs_dict = self._build_dict(self.obs_keys, obs_inputs)
            obs_dict = self._process_obs(obs_dict)

            goal_dict = None
            if self.is_goal_conditioned:
                goal_inputs = inputs[idx: idx + goal_count]
                idx += goal_count
                goal_dict = self._build_dict(self.goal_keys, goal_inputs)
                goal_dict = self._process_goal(goal_dict)

            if self.is_recurrent:
                h0, c0 = inputs[idx], inputs[idx + 1]
                actions, (h1, c1) = self.actor_net(
                    obs_dict,
                    goal_dict=goal_dict,
                    rnn_init_state=(h0, c0),
                    return_state=True,
                )
                actions = self._postprocess_actions(actions)
                return actions, h1, c1

            output = self.actor_net(obs_dict, goal_dict=goal_dict)
            if isinstance(output, torch.Tensor):
                return self._postprocess_actions(output)
            if isinstance(output, dict):
                return self._postprocess_actions(output)
            if isinstance(output, (list, tuple)) and output:
                processed = self._postprocess_actions(output[0])
                rest = output[1:]
                if isinstance(output, list):
                    return [processed, *rest]
                return (processed, *rest)
            return output
    wrapper = OnnxPolicy().cpu().eval()
    return wrapper, obs_keys, goal_keys, is_recurrent, obs_shapes, goal_shapes, rnn_cfg


def create_dummy_inputs(
    obs_shapes: Mapping[str, Sequence[int]],
    goal_shapes: Mapping[str, Sequence[int]],
    *,
    is_recurrent: bool,
    rnn_cfg: Mapping[str, Any],
) -> Tuple[List[torch.Tensor], List[str], List[str]]:
    batch = 1
    seq = 1 if is_recurrent else None

    def _raw_shape(key: str, processed_shape: Sequence[int]) -> Sequence[int]:
        modality_lookup = getattr(ObsUtils, "OBS_KEYS_TO_MODALITIES", None)
        modality = modality_lookup.get(key) if isinstance(modality_lookup, Mapping) else None
        if modality in ("rgb", "depth", "scan"):
            sample = torch.zeros((1, *processed_shape), dtype=torch.float32)
            try:
                raw = ObsUtils.unprocess_obs(sample, obs_key=key)
            except Exception:
                return tuple(processed_shape)
            raw_shape = tuple(int(dim) for dim in raw.shape)
            if raw_shape and raw_shape[0] == 1:
                raw_shape = raw_shape[1:]
            return raw_shape or tuple(processed_shape)
        return tuple(processed_shape)

    def make_tensor(key: str, shape: Sequence[int]) -> torch.Tensor:
        raw_shape = _raw_shape(key, shape)
        size = (batch, *raw_shape) if not is_recurrent else (batch, seq, *raw_shape)
        return torch.zeros(size, dtype=torch.float32)

    obs_tensors = [make_tensor(key, obs_shapes[key]) for key in obs_shapes]
    goal_tensors = [make_tensor(key, goal_shapes[key]) for key in goal_shapes]

    inputs: List[torch.Tensor] = obs_tensors + goal_tensors
    input_names = [f"obs::{key}" for key in obs_shapes]
    input_names += [f"goal::{key}" for key in goal_shapes]

    if is_recurrent:
        num_layers = int(rnn_cfg.get("num_layers", 1))
        hidden_dim = int(rnn_cfg.get("hidden_dim"))
        inputs.extend([
            torch.zeros((num_layers, batch, hidden_dim), dtype=torch.float32),
            torch.zeros((num_layers, batch, hidden_dim), dtype=torch.float32),
        ])
        input_names += ["h0", "c0"]

    output_names = ["actions"] + (["h1", "c1"] if is_recurrent else [])
    return inputs, input_names, output_names


def export_to_onnx(
    model: torch.nn.Module,
    dummy_inputs: Sequence[torch.Tensor],
    input_names: Sequence[str],
    output_names: Sequence[str],
    onnx_path: pathlib.Path,
    *,
    opset_version: int = 17,
) -> None:
    torch.onnx.export(
        model,
        tuple(dummy_inputs),
        pathlib.Path(onnx_path).expanduser().as_posix(),
        input_names=list(input_names),
        output_names=list(output_names),
        opset_version=opset_version,
        do_constant_folding=True,
    )


# ---------------------------------------------------------------------------
# Metadata helpers
# ---------------------------------------------------------------------------

def make_metadata(
    *,
    ckpt_path: pathlib.Path,
    onnx_path: pathlib.Path,
    input_names: Sequence[str],
    output_names: Sequence[str],
    obs_keys: Sequence[str],
    goal_keys: Sequence[str],
    is_recurrent: bool,
    obs_stats_filename: Optional[str],
    action_stats_filename: Optional[str],
    needs_postprocessing: bool,
    export_directory: pathlib.Path,
    requires_preprocessing: bool = False,
    requires_postprocessing: Optional[bool] = None,
) -> Dict[str, Any]:
    export_dir = pathlib.Path(export_directory).expanduser()
    onnx_path = pathlib.Path(onnx_path).expanduser()
    if requires_postprocessing is None:
        requires_postprocessing = bool(needs_postprocessing)
    return {
        "model_name": pathlib.Path(ckpt_path).stem,
        "checkpoint_file": pathlib.Path(ckpt_path).name,
        "model_file": onnx_path.name,
        "model_path": onnx_path.as_posix(),
        "export_directory": export_dir.as_posix(),
        "inputs": list(input_names),
        "outputs": list(output_names),
        "observation_keys": list(obs_keys),
        "goal_keys": list(goal_keys),
        "is_recurrent": bool(is_recurrent),
        "requires_preprocessing": bool(requires_preprocessing),
        "requires_postprocessing": bool(requires_postprocessing),
        "normalization_files": {
            "observation": obs_stats_filename,
            "action": action_stats_filename,
        },
    }


def write_metadata(metadata: Mapping[str, Any], path: pathlib.Path) -> None:
    pathlib.Path(path).expanduser().write_text(json.dumps(metadata, indent=2))




