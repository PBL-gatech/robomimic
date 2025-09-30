#!/usr/bin/env python
"""
Export a Robomimic policy checkpoint to ONNX Runtime format.

Usage (example):
    # Auto-find config and latest model in folder:
    python export_to_onnx.py --folder bc_patcherBot_trained_models_HEK_v0_015/v0_015/20250506223704/

    # Or specify files manually (original approach):
    python export_to_onnx.py --ckpt model_epoch_485.pth --config config.json --out policy.onnx
"""
import argparse, torch, json, pathlib, glob, re, copy
from collections import OrderedDict

from robomimic.utils.file_utils import policy_from_checkpoint
from robomimic.utils import obs_utils as ObsUtils
import robomimic.utils.tensor_utils as TensorUtils


# df_path = r"C:\Users\sa-forest\Documents\GitHub\robomimic\df_patcherBot\PipetteFinding\v0_004\20250927024406"
# bc_path = r"C:\Users\sa-forest\Documents\GitHub\robomimic\bc_patcherBot\PipetteFinding\v0_003\20250926165240"
# bc_path = r"C:\Users\sa-forest\Documents\GitHub\robomimic\bc_patcherBot\PipetteFinding\v0_001\20250925220945"
# bc_path =r"C:\Users\sa-forest\Documents\GitHub\robomimic\bc_patcherBot\PipetteFinding\v0_005\20250928193451"
# bc_path =r"C:\Users\sa-forest\Documents\GitHub\robomimic\bc_patcherBot\NeuronHunting\v0_041\20250928212815"
# bc_path = r"C:\Users\sa-forest\Documents\GitHub\robomimic\bc_patcherBot\PipetteFinding\v0_007\20250929154731"
bc_path = r"C:\Users\sa-forest\Documents\GitHub\robomimic\bc_patcherBot\PipetteFinding\v0_009\20250930123649"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", help="Path to .pth checkpoint")
    parser.add_argument("--config", help="Path to config.json")
    parser.add_argument("--out", help="Output ONNX file")
    parser.add_argument("--folder", help="Path to folder containing checkpoint and config", default=bc_path)
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"], help="Device to use")
    args = parser.parse_args()
    print(f"[*] Parsed arguments: {args}")
    return args


def find_paths(folder):
    folder_path = pathlib.Path(folder).expanduser()
    # config search
    configs = list(folder_path.rglob('config.json'))
    if not configs:
        raise FileNotFoundError(f"Config file not found under {folder_path}")
    config_path = configs[0]
    # checkpoint search
    ckpts = glob.glob(str(folder_path / '**' / 'model_epoch_*.pth'), recursive=True)
    if not ckpts:
        raise FileNotFoundError(f"No checkpoint files found under {folder_path}")
    ckpts = sorted(ckpts, key=lambda x: int(re.search(r'model_epoch_(\d+)', x).group(1)))
    ckpt_path = pathlib.Path(ckpts[-1])
    onnx_path = folder_path / f"{ckpt_path.stem}.onnx"
    print(f"[*] Completed path search: config={config_path}, ckpt={ckpt_path}, onnx={onnx_path}")
    return config_path, ckpt_path, onnx_path


def load_policy(ckpt_path, device):
    print(f"[*] Loading policy from checkpoint {ckpt_path}")
    policy, ckpt_dict = policy_from_checkpoint(
        ckpt_path=ckpt_path,
        device=torch.device(device),
        verbose=False,
    )
    print("[*] Policy loaded")
    return policy, ckpt_dict


def inspect_policy(ckpt_path, device):
    """Utility to inspect a checkpoint when debugging."""
    print(f"[*] Inspecting checkpoint: {ckpt_path}")
    raw_ckpt = torch.load(ckpt_path, map_location=device)
    print("[*] Raw checkpoint keys:", list(raw_ckpt.keys()))
    if "model" in raw_ckpt:
        print("[*]   model sub-dict keys:", list(raw_ckpt["model"].keys()))

    policy, ckpt_dict = policy_from_checkpoint(
        ckpt_path=ckpt_path,
        device=torch.device(device),
        verbose=False,
    )
    print("[*] Extracted ckpt_dict top-level keys:", list(ckpt_dict.keys()))
    print("[*] Policy public attributes:", [k for k in dir(policy) if not k.startswith("_")])
    return policy, ckpt_dict


def load_config(config_path):
    """Load JSON config and print relevant sections."""
    cfg = json.load(open(config_path, "r"))
    print("[load_config] obs modalities:", cfg["observation"]["modalities"]["obs"])
    algo_section = cfg.get("algo", {})
    rnn_cfg = algo_section.get("rnn") if isinstance(algo_section, dict) else None
    if rnn_cfg is not None:
        print("[load_config] rnn settings:", rnn_cfg)
    else:
        print("[load_config] rnn settings: <none>")
    return cfg


# ----------------------------
# Normalization stats (actions)
# ----------------------------

def _to_1d_np(x):
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    import numpy as np
    return np.asarray(x, dtype=np.float32).reshape(-1)


def extract_action_normalization(ckpt_dict):
    """
    Build action normalization dict compatible with ObsUtils.unnormalize_dict.
    Returns {"actions": {"offset": torch.Tensor(1,1,D), "scale": torch.Tensor(1,1,D)}} or None.
    Accepts offset/scale or mean/std even when nested under intermediate keys.
    """
    def _find_stats(node):
        if not isinstance(node, dict):
            return None
        if ("offset" in node and "scale" in node) or ("mean" in node and "std" in node):
            return node
        for value in node.values():
            found = _find_stats(value)
            if found is not None:
                return found
        return None

    stats = None
    if isinstance(ckpt_dict, dict):
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
            if sub is None:
                stats = _find_stats(node)
            elif isinstance(node, dict):
                stats = _find_stats(node.get(sub))
            if stats is not None:
                break
        if stats is None:
            stats = _find_stats(ckpt_dict)

    if stats is None:
        print("[extract_action_normalization] No action normalization stats found")
        return None

    if "offset" in stats and "scale" in stats:
        offset, scale = stats["offset"], stats["scale"]
    elif "mean" in stats and "std" in stats:
        offset, scale = stats["mean"], stats["std"]
    else:
        print("[extract_action_normalization] Missing offset/scale or mean/std")
        return None

    offset = _to_1d_np(offset)
    scale = _to_1d_np(scale)
    dim = int(offset.shape[0])

    offset_t = torch.from_numpy(offset).view(1, dim)
    scale_t = torch.from_numpy(scale).view(1, dim)

    return {"actions": {"offset": offset_t, "scale": scale_t}}




# ----------------------------
# Normalization stats (observations)
# ----------------------------


def extract_obs_normalization(policy, obs_keys, goal_keys):
    """Convert observation normalization stats on the RolloutPolicy into torch tensors."""
    stats = getattr(policy, "obs_normalization_stats", None)
    if stats is None:
        print("[extract_obs_normalization] Policy does not expose observation normalization stats")
        return OrderedDict()

    ordered_keys = []
    for key in list(obs_keys) + list(goal_keys):
        if key not in ordered_keys:
            ordered_keys.append(key)

    filtered = OrderedDict((key, stats[key]) for key in ordered_keys if key in stats)
    missing = [key for key in ordered_keys if key not in stats]

    if not filtered:
        if missing:
            print(f"[extract_obs_normalization] Missing stats for keys: {sorted(missing)}")
        print("[extract_obs_normalization] No observation keys matched normalization stats")
        return OrderedDict()

    tensor_stats = TensorUtils.to_float(TensorUtils.to_tensor(filtered))

    if missing:
        print(f"[extract_obs_normalization] Missing stats for keys: {sorted(missing)}")
    print(f"[extract_obs_normalization] Loaded stats for keys: {list(tensor_stats.keys())}")
    return tensor_stats



def _unnormalize_actions_tensor(actions, offset, scale):
    """Broadcast action normalization buffers to match actions and apply affine transform."""
    if offset is None or scale is None:
        return actions
    device = actions.device
    dtype = actions.dtype
    offset_t = offset.to(device=device, dtype=dtype)
    scale_t = scale.to(device=device, dtype=dtype)
    if offset_t.dim() < actions.dim():
        view_shape = (1,) * (actions.dim() - offset_t.dim()) + tuple(offset_t.shape)
        offset_t = offset_t.view(*view_shape)
        scale_t = scale_t.view(*view_shape)
    offset_t = offset_t.expand_as(actions)
    scale_t = scale_t.expand_as(actions)
    return actions * scale_t + offset_t


def _is_tracing_or_onnx_export():
    """Return True when torch tracing or ONNX export is active."""
    if torch.jit.is_tracing():
        return True
    is_in_onnx_export = getattr(torch.onnx, "is_in_onnx_export", None)
    if callable(is_in_onnx_export):
        try:
            if is_in_onnx_export():
                return True
        except Exception:
            pass
    return False


def _apply_tracing_safe_overrides():
    """Patch selected robomimic utilities to avoid tracer warnings during ONNX export."""
    if getattr(_apply_tracing_safe_overrides, "_applied", False):
        return
    _apply_tracing_safe_overrides._applied = True

    from robomimic.utils import obs_utils as _ObsUtils
    import robomimic.utils.tensor_utils as _TensorUtils
    from robomimic.models import base_nets as _BaseNets
    from robomimic.models import obs_core as _ObsCore
    import torch
    import numpy as _np

    original_center_crop = _ObsUtils.center_crop

    def _center_crop_safe(im, t_h, t_w):
        if not _is_tracing_or_onnx_export():
            return original_center_crop(im, t_h, t_w)
        crop_h = (im.shape[-3] - t_h) // 2
        crop_w = (im.shape[-2] - t_w) // 2
        return im[..., crop_h:crop_h + t_h, crop_w:crop_w + t_w, :]

    _ObsUtils.center_crop = _center_crop_safe

    original_process_frame = _ObsUtils.process_frame

    def _process_frame_safe(frame, channel_dim, scale):
        if not _is_tracing_or_onnx_export():
            return original_process_frame(frame, channel_dim, scale)
        frame = _TensorUtils.to_float(frame)
        if scale is not None:
            frame = frame / scale
            frame = frame.clamp(0.0, 1.0)
        return _ObsUtils.batch_image_hwc_to_chw(frame)

    _ObsUtils.process_frame = _process_frame_safe

    original_normalize_dict = _ObsUtils.normalize_dict

    def _normalize_dict_safe(data_dict, normalization_stats):
        if not _is_tracing_or_onnx_export():
            return original_normalize_dict(data_dict, normalization_stats)
        for key, value in data_dict.items():
            stats = normalization_stats[key]
            offset = stats["offset"][0]
            scale = stats["scale"][0]
            while offset.dim() < value.dim():
                offset = offset.unsqueeze(0)
                scale = scale.unsqueeze(0)
            data_dict[key] = (value - offset) / scale
        return data_dict

    _ObsUtils.normalize_dict = _normalize_dict_safe

    original_unnormalize_dict = _ObsUtils.unnormalize_dict

    def _unnormalize_dict_safe(data_dict, normalization_stats):
        if not _is_tracing_or_onnx_export():
            return original_unnormalize_dict(data_dict, normalization_stats)
        for key, value in data_dict.items():
            stats = normalization_stats[key]
            offset = stats["offset"]
            scale = stats["scale"]
            while offset.dim() < value.dim():
                offset = offset.unsqueeze(0)
                scale = scale.unsqueeze(0)
            data_dict[key] = (value * scale) + offset
        return data_dict

    _ObsUtils.unnormalize_dict = _unnormalize_dict_safe

    original_convbase_forward = _BaseNets.ConvBase.forward

    def _convbase_forward_safe(self, inputs):
        if _is_tracing_or_onnx_export():
            return self.nets(inputs)
        return original_convbase_forward(self, inputs)

    _BaseNets.ConvBase.forward = _convbase_forward_safe

    if hasattr(_BaseNets, "SpatialSoftmax"):
        original_spatial_forward = _BaseNets.SpatialSoftmax.forward

        def _spatial_forward_safe(self, feature):
            if not _is_tracing_or_onnx_export():
                return original_spatial_forward(self, feature)
            if self.nets is not None:
                feature = self.nets(feature)
            feature = feature.reshape(-1, self._in_h * self._in_w)
            attention = torch.softmax(feature / self.temperature, dim=-1)
            expected_x = torch.sum(self.pos_x * attention, dim=1, keepdim=True)
            expected_y = torch.sum(self.pos_y * attention, dim=1, keepdim=True)
            expected_xy = torch.cat([expected_x, expected_y], 1)
            feature_keypoints = expected_xy.view(-1, self._num_kp, 2)
            if self.training:
                feature_keypoints = feature_keypoints + torch.randn_like(feature_keypoints) * self.noise_std
            if self.output_variance:
                expected_xx = torch.sum(self.pos_x * self.pos_x * attention, dim=1, keepdim=True)
                expected_yy = torch.sum(self.pos_y * self.pos_y * attention, dim=1, keepdim=True)
                expected_xy_cov = torch.sum(self.pos_x * self.pos_y * attention, dim=1, keepdim=True)
                var_x = expected_xx - expected_x * expected_x
                var_y = expected_yy - expected_y * expected_y
                var_xy = expected_xy_cov - expected_x * expected_y
                covar = torch.cat([var_x, var_xy, var_xy, var_y], 1).reshape(-1, self._num_kp, 2, 2)
                feature_keypoints = (feature_keypoints, covar)
            return feature_keypoints

        _BaseNets.SpatialSoftmax.forward = _spatial_forward_safe

    original_visualcore_forward = _ObsCore.VisualCore.forward

    def _visualcore_forward_safe(self, inputs):
        if _is_tracing_or_onnx_export():
            return _BaseNets.ConvBase.forward(self, inputs)
        return original_visualcore_forward(self, inputs)

    _ObsCore.VisualCore.forward = _visualcore_forward_safe

    original_moveaxis = torch.Tensor.moveaxis

    def _moveaxis_safe(tensor, source, destination):
        if not _is_tracing_or_onnx_export():
            return original_moveaxis(tensor, source, destination)
        ndim = tensor.dim()
        src_axes = _np.atleast_1d(_np.array(source, dtype=int)).tolist()
        dst_axes = _np.atleast_1d(_np.array(destination, dtype=int)).tolist()
        src_axes = [axis + ndim if axis < 0 else axis for axis in src_axes]
        dst_axes = [axis + ndim if axis < 0 else axis for axis in dst_axes]
        if len(src_axes) != len(dst_axes):
            raise ValueError('source and destination must have the same number of axes')
        perm = [axis for axis in range(ndim) if axis not in src_axes]
        for dest_axis, src_axis in sorted(zip(dst_axes, src_axes)):
            perm.insert(dest_axis, src_axis)
        return tensor.permute(*perm)

    torch.Tensor.moveaxis = _moveaxis_safe


def wrap_policy(policy, config, action_norm_stats, ckpt_dict):
    """
    Dispatch to the appropriate ONNX wrapper based on the checkpoint algorithm.
    """
    policy_obj = policy[0] if isinstance(policy, tuple) else policy
    algo_name = str(ckpt_dict.get("algo_name", ""))
    algo_name = algo_name.lower()
    if algo_name == "diffusion_policy":
        print("[wrap_policy] Detected diffusion policy checkpoint")
        return wrap_diffusion_policy(policy_obj, action_norm_stats or {})
    print(f"[wrap_policy] Using standard wrapper for algo='{algo_name or 'unknown'}'")
    return wrap_standard_policy(policy_obj, config, action_norm_stats or {})


def wrap_standard_policy(policy_obj, config, action_norm_stats):
    """Existing BC-style wrapper supporting feedforward and RNN actors."""
    import torch.nn as nn

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

    if hasattr(actor_net, "obs_shapes"):
        obs_shapes = OrderedDict(actor_net.obs_shapes)
    elif hasattr(inner, "obs_shapes"):
        obs_shapes = OrderedDict(inner.obs_shapes)
    else:
        raise AttributeError("Could not locate obs_shapes on policy. Check your Robomimic version.")

    goal_shapes = OrderedDict()
    if hasattr(actor_net, "goal_shapes") and actor_net.goal_shapes:
        goal_shapes = OrderedDict(actor_net.goal_shapes)
    elif hasattr(inner, "goal_shapes") and inner.goal_shapes:
        goal_shapes = OrderedDict(inner.goal_shapes)

    obs_keys = list(obs_shapes.keys())
    goal_keys = list(goal_shapes.keys())
    algo_cfg = config.get("algo", {}) if isinstance(config, dict) else {}
    rnn_cfg = algo_cfg.get("rnn", {}) if isinstance(algo_cfg, dict) else {}
    is_recurrent = bool(rnn_cfg.get("enabled", False))

    obs_norm_stats = extract_obs_normalization(policy_obj, obs_keys, goal_keys)
    action_norm = action_norm_stats.get("actions") if action_norm_stats else None

    class OnnxPolicy(nn.Module):
        def __init__(self, actor_net, obs_keys, goal_keys, obs_shapes, goal_shapes, is_recurrent, obs_norm_stats, action_norm):
            super().__init__()
            self.actor_net = actor_net
            self.obs_keys = obs_keys
            self.goal_keys = goal_keys
            self.is_goal_conditioned = len(goal_keys) > 0
            self.is_recurrent = is_recurrent
            self.obs_shapes = obs_shapes
            self.goal_shapes = goal_shapes
            self.rnn = getattr(actor_net, "rnn", None)
            self.requires_action_cache = False
            self.observation_horizon = 1
            self.output_names = ["actions"] + (["h1", "c1"] if self.is_recurrent else [])

            self.obs_norm_stats = OrderedDict()
            if obs_norm_stats:
                for idx, (key, stats) in enumerate(obs_norm_stats.items()):
                    offset = stats["offset"]
                    scale = stats["scale"]
                    target_dim = len(self.obs_shapes[key]) + 1
                    while offset.dim() < target_dim:
                        offset = TensorUtils.unsqueeze(offset, 0)
                        scale = TensorUtils.unsqueeze(scale, 0)
                    offset_name = f"obs_offset_{idx}"
                    scale_name = f"obs_scale_{idx}"
                    self.register_buffer(offset_name, offset.to(torch.float32))
                    self.register_buffer(scale_name, scale.to(torch.float32))
                    self.obs_norm_stats[key] = {
                        "offset": getattr(self, offset_name),
                        "scale": getattr(self, scale_name),
                    }

            if action_norm is not None:
                self.register_buffer("act_offset", action_norm["offset"].to(torch.float32))
                self.register_buffer("act_scale", action_norm["scale"].to(torch.float32))
            else:
                self.act_offset = None
                self.act_scale = None

        @staticmethod
        def _extract_actions(actions):
            if isinstance(actions, dict):
                return actions.get("actions", next(iter(actions.values())))
            return actions

        def _unnormalize_actions(self, actions):
            return _unnormalize_actions_tensor(actions, self.act_offset, self.act_scale)

        def _build_obs_dict(self, keys, tensors):
            return OrderedDict((k, v) for k, v in zip(keys, tensors))

        def _apply_obs_normalization(self, tensors):
            if not self.obs_norm_stats:
                return tensors
            sub_stats = OrderedDict((k, self.obs_norm_stats[k]) for k in tensors if k in self.obs_norm_stats)
            if sub_stats:
                ObsUtils.normalize_dict(tensors, normalization_stats=sub_stats)
            return tensors

        def _process_obs_group(self, tensors):
            if not tensors:
                return tensors
            processed = TensorUtils.to_float(tensors)
            processed = ObsUtils.process_obs_dict(processed)
            processed = self._apply_obs_normalization(processed)
            return processed

        def _process_goal_group(self, tensors):
            if not tensors:
                return tensors
            processed = self._process_obs_group(tensors)
            sanitized = OrderedDict()
            for key, tensor in processed.items():
                target_shape = self.goal_shapes.get(key)
                if target_shape is None:
                    sanitized[key] = tensor
                    continue
                target_ndim = len(target_shape) + 1  # batch dim + feature dims
                while tensor.dim() > target_ndim and tensor.shape[1] == 1:
                    tensor = tensor.squeeze(1)
                sanitized[key] = tensor
            return sanitized

        def forward(self, *tensors):
            idx = 0
            obs_count = len(self.obs_keys)
            goal_count = len(self.goal_keys)

            obs_inputs = tensors[idx: idx + obs_count]
            idx += obs_count
            obs_dict = self._build_obs_dict(self.obs_keys, obs_inputs)
            obs_dict = self._process_obs_group(obs_dict)

            goal_dict = None
            if self.is_goal_conditioned:
                goal_inputs = tensors[idx: idx + goal_count]
                idx += goal_count
                goal_dict = self._build_obs_dict(self.goal_keys, goal_inputs)
                goal_dict = self._process_goal_group(goal_dict)

            if self.is_recurrent:
                h0, c0 = tensors[idx], tensors[idx + 1]
                actions, (h1, c1) = self.actor_net(
                    obs_dict,
                    goal_dict=goal_dict,
                    rnn_init_state=(h0, c0),
                    return_state=True,
                )
                actions = self._extract_actions(actions)
                actions = self._unnormalize_actions(actions)
                return actions, h1, c1

            actions = self.actor_net(obs_dict, goal_dict=goal_dict)
            actions = self._extract_actions(actions)
            actions = self._unnormalize_actions(actions)
            return actions

    wrapper = OnnxPolicy(actor_net, obs_keys, goal_keys, obs_shapes, goal_shapes, is_recurrent, obs_norm_stats, action_norm).cpu().eval()
    print(f"[wrap_policy] Wrapper ready - obs: {obs_keys} | goal: {goal_keys} | recurrent: {is_recurrent}")
    if obs_norm_stats:
        print("[wrap_policy] Observation normalization baked into wrapper")
    else:
        print("[wrap_policy] No observation normalization stats found; inputs assumed pre-normalized")
    if action_norm is not None:
        print("[wrap_policy] Action unnormalization buffers registered")
    else:
        print("[wrap_policy] No action normalization found; ONNX will output normalized actions ([-1, 1])")
    return wrapper, obs_keys, goal_keys, is_recurrent


def wrap_diffusion_policy(policy_obj, action_norm_stats):
    """Wrapper for diffusion policies that manages action horizons via a cache."""
    import torch
    import torch.nn as nn

    inner = policy_obj.policy
    if not hasattr(inner, "algo_config"):
        raise AttributeError("Diffusion policy is missing algo_config")

    algo_cfg = inner.algo_config
    horizon_cfg = getattr(algo_cfg, "horizon", None)
    if horizon_cfg is None:
        raise AttributeError("Diffusion policy config missing horizon settings")

    action_horizon = int(horizon_cfg.action_horizon)
    observation_horizon = int(horizon_cfg.observation_horizon)
    prediction_horizon = int(horizon_cfg.prediction_horizon)

    obs_shapes = OrderedDict(inner.obs_shapes)
    goal_shapes = OrderedDict(getattr(inner, "goal_shapes", OrderedDict()))
    obs_keys = list(obs_shapes.keys())
    goal_keys = list(goal_shapes.keys())

    obs_norm_stats = extract_obs_normalization(policy_obj, obs_keys, goal_keys)
    action_norm = action_norm_stats.get("actions") if action_norm_stats else None

    if inner.ema is not None:
        policy_module = inner.ema.averaged_model["policy"]
    else:
        policy_module = inner.nets["policy"]

    obs_encoder = policy_module["obs_encoder"]
    noise_pred_net = policy_module["noise_pred_net"]
    noise_scheduler = copy.deepcopy(inner.noise_scheduler)
    if noise_scheduler is None:
        raise ValueError("Diffusion policy missing noise scheduler")

    if algo_cfg.ddpm.enabled:
        num_inference_timesteps = int(algo_cfg.ddpm.num_inference_timesteps)
    elif algo_cfg.ddim.enabled:
        num_inference_timesteps = int(algo_cfg.ddim.num_inference_timesteps)
    else:
        raise ValueError("Unsupported diffusion scheduler configuration")
    noise_scheduler.set_timesteps(num_inference_timesteps)
    timesteps = noise_scheduler.timesteps
    if not isinstance(timesteps, torch.Tensor):
        timesteps = torch.tensor(list(timesteps), dtype=torch.long)
    else:
        timesteps = timesteps.clone().to(torch.long)
    timesteps_list = tuple(int(t) for t in timesteps.view(-1).tolist())

    class DiffusionOnnxPolicy(nn.Module):
        def __init__(self, obs_encoder, noise_pred_net, noise_scheduler, timesteps, timesteps_list, obs_keys, goal_keys, obs_shapes, goal_shapes, obs_norm_stats, action_norm, action_horizon, observation_horizon, prediction_horizon, action_dim, num_inference_timesteps):
            super().__init__()
            self.obs_encoder = obs_encoder
            self.noise_pred_net = noise_pred_net
            self.noise_scheduler = noise_scheduler
            self.register_buffer("timesteps", timesteps)
            self.timesteps_list = timesteps_list
            self.num_inference_timesteps = num_inference_timesteps

            self.obs_keys = obs_keys
            self.goal_keys = goal_keys
            self.obs_shapes = obs_shapes
            self.goal_shapes = goal_shapes
            self.requires_action_cache = True
            self.is_recurrent = False
            self.output_names = ["action", "action_cache", "cache_index"]

            self.action_horizon = action_horizon
            self.observation_horizon = observation_horizon
            self.prediction_horizon = prediction_horizon
            self.action_dim = action_dim
            self.start_index = max(observation_horizon - 1, 0)

            self.obs_norm_stats = OrderedDict()
            if obs_norm_stats:
                for idx, (key, stats) in enumerate(obs_norm_stats.items()):
                    offset = stats["offset"]
                    scale = stats["scale"]
                    target_dim = len(self.obs_shapes[key]) + 1
                    while offset.dim() < target_dim:
                        offset = TensorUtils.unsqueeze(offset, 0)
                        scale = TensorUtils.unsqueeze(scale, 0)
                    offset_name = f"obs_offset_{idx}"
                    scale_name = f"obs_scale_{idx}"
                    self.register_buffer(offset_name, offset.to(torch.float32))
                    self.register_buffer(scale_name, scale.to(torch.float32))
                    self.obs_norm_stats[key] = {
                        "offset": getattr(self, offset_name),
                        "scale": getattr(self, scale_name),
                    }

            if action_norm is not None:
                self.register_buffer("act_offset", action_norm["offset"].to(torch.float32))
                self.register_buffer("act_scale", action_norm["scale"].to(torch.float32))
            else:
                self.act_offset = None
                self.act_scale = None

        def _build_obs_dict(self, keys, tensors):
            return OrderedDict((k, v) for k, v in zip(keys, tensors))

        def _apply_obs_normalization(self, tensors):
            if not self.obs_norm_stats:
                return tensors
            sub_stats = OrderedDict((k, self.obs_norm_stats[k]) for k in tensors if k in self.obs_norm_stats)
            if sub_stats:
                ObsUtils.normalize_dict(tensors, normalization_stats=sub_stats)
            return tensors

        def _process_obs_group(self, tensors):
            if not tensors:
                return tensors
            processed = TensorUtils.to_float(tensors)
            processed = ObsUtils.process_obs_dict(processed)
            processed = self._apply_obs_normalization(processed)
            return processed

        def _process_goal_group(self, tensors):
            if not tensors:
                return tensors
            processed = self._process_obs_group(tensors)
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

        def _unnormalize_actions(self, actions):
            return _unnormalize_actions_tensor(actions, self.act_offset, self.act_scale)

        def _run_diffusion(self, obs_dict, goal_dict):
            inputs = {"obs": obs_dict, "goal": goal_dict}
            for key in self.obs_shapes:
                tensor = inputs["obs"][key]
                if tensor.ndim - 1 == len(self.obs_shapes[key]):
                    inputs["obs"][key] = tensor.unsqueeze(1)
                elif not _is_tracing_or_onnx_export():
                    assert inputs["obs"][key].ndim - 2 == len(self.obs_shapes[key])
            obs_features = TensorUtils.time_distributed(inputs, self.obs_encoder, inputs_as_kwargs=True)
            if not _is_tracing_or_onnx_export():
                assert obs_features.ndim == 3
            batch = obs_features.shape[0]
            obs_cond = obs_features.flatten(start_dim=1)

            noise = torch.randn((batch, self.prediction_horizon, self.action_dim), device=obs_cond.device, dtype=obs_cond.dtype)
            naction = noise

            self.noise_scheduler.set_timesteps(self.num_inference_timesteps)
            for timestep in self.timesteps_list:
                noise_pred = self.noise_pred_net(sample=naction, timestep=timestep, global_cond=obs_cond)
                naction = self.noise_scheduler.step(model_output=noise_pred, timestep=timestep, sample=naction).prev_sample

            end = self.start_index + self.action_horizon
            return naction[:, self.start_index:end]

        def forward(self, *tensors):
            idx = 0
            obs_inputs = tensors[idx: idx + len(self.obs_keys)]
            idx += len(self.obs_keys)
            obs_dict = self._build_obs_dict(self.obs_keys, obs_inputs)
            obs_dict = self._process_obs_group(obs_dict)

            goal_dict = None
            if len(self.goal_keys) > 0:
                goal_inputs = tensors[idx: idx + len(self.goal_keys)]
                idx += len(self.goal_keys)
                goal_dict = self._build_obs_dict(self.goal_keys, goal_inputs)
                goal_dict = self._process_goal_group(goal_dict)

            cached_actions = tensors[idx]
            cache_index = tensors[idx + 1]

            if cached_actions.dim() == 2:
                cached_actions = cached_actions.unsqueeze(0)
            cache_index = cache_index.view(cached_actions.shape[0]).long()
            cache_index = torch.clamp(cache_index, min=0)
            cached_actions = TensorUtils.to_float(cached_actions)

            new_actions = self._run_diffusion(obs_dict, goal_dict)
            new_actions = self._unnormalize_actions(new_actions)

            needs_refresh = cache_index >= self.action_horizon
            refresh_mask = needs_refresh.view(-1, 1, 1)
            combined_actions = torch.where(refresh_mask, new_actions, cached_actions)

            gather_index = torch.where(needs_refresh, torch.zeros_like(cache_index), torch.clamp(cache_index, max=self.action_horizon - 1))
            gather_index = gather_index.view(-1, 1, 1).repeat(1, 1, self.action_dim)
            action = torch.gather(combined_actions, 1, gather_index).squeeze(1)

            next_index = torch.where(needs_refresh, torch.ones_like(cache_index), cache_index + 1)
            next_index = torch.clamp(next_index, max=self.action_horizon)

            return action, combined_actions, next_index

    action_dim = int(inner.ac_dim)
    wrapper = DiffusionOnnxPolicy(
        obs_encoder=obs_encoder,
        noise_pred_net=noise_pred_net,
        noise_scheduler=noise_scheduler,
        timesteps=timesteps,
        timesteps_list=timesteps_list,
        obs_keys=obs_keys,
        goal_keys=goal_keys,
        obs_shapes=obs_shapes,
        goal_shapes=goal_shapes,
        obs_norm_stats=obs_norm_stats,
        action_norm=action_norm,
        action_horizon=action_horizon,
        observation_horizon=observation_horizon,
        prediction_horizon=prediction_horizon,
        action_dim=action_dim,
        num_inference_timesteps=num_inference_timesteps,
    ).cpu().eval()

    print(f"[wrap_policy] Diffusion wrapper ready - obs: {obs_keys} | goal: {goal_keys} | action_horizon: {action_horizon}")
    if obs_norm_stats:
        print("[wrap_policy] Observation normalization baked into diffusion wrapper")
    else:
        print("[wrap_policy] Diffusion wrapper expects pre-normalized observations")
    if action_norm is not None:
        print("[wrap_policy] Action unnormalization enabled for diffusion wrapper")
    else:
        print("[wrap_policy] Diffusion wrapper will output normalized actions ([-1, 1])")

    is_recurrent = False
    return wrapper, obs_keys, goal_keys, is_recurrent

def create_dummy_inputs(wrapper, obs_keys, goal_keys, is_recurrent, config):
    '''
    Build fixed-size zero tensors for export. All dimensions-including batch
    and sequence length-are constant, so OpenCV DNN will load the file.
    '''
    import torch

    is_recurrent = bool(getattr(wrapper, "is_recurrent", is_recurrent))
    batch = 1
    seq = max(1, int(getattr(wrapper, "observation_horizon", 1)))

    def _raw_shape(key, processed_shape):
        modality = ObsUtils.OBS_KEYS_TO_MODALITIES.get(key) if ObsUtils.OBS_KEYS_TO_MODALITIES else None
        if modality in ("rgb", "depth", "scan"):
            try:
                sample = torch.zeros((1, *processed_shape), dtype=torch.float32)
                raw = ObsUtils.unprocess_obs(sample, obs_key=key)
                raw_shape = tuple(int(dim) for dim in raw.shape)
                if len(raw_shape) > 0 and raw_shape[0] == 1:
                    raw_shape = raw_shape[1:]
                return raw_shape or processed_shape
            except Exception:
                return processed_shape
        return processed_shape

    obs_tensors = [
        torch.zeros((batch, seq, *_raw_shape(k, wrapper.obs_shapes[k])), dtype=torch.float32)
        for k in obs_keys
    ]
    goal_tensors = [
        torch.zeros((batch, *_raw_shape(k, wrapper.goal_shapes[k])), dtype=torch.float32)
        for k in goal_keys
    ]

    dummy_inputs = obs_tensors + goal_tensors

    if is_recurrent:
        num_layers = config["algo"]["rnn"]["num_layers"]
        hidden_size = config["algo"]["rnn"]["hidden_dim"]
        dummy_inputs.extend([
            torch.zeros((num_layers, batch, hidden_size), dtype=torch.float32),  # h0
            torch.zeros((num_layers, batch, hidden_size), dtype=torch.float32),  # c0
        ])

    if getattr(wrapper, "requires_action_cache", False):
        action_horizon = int(getattr(wrapper, "action_horizon", 1))
        action_dim = int(getattr(wrapper, "action_dim", 1))
        cache = torch.zeros((batch, action_horizon, action_dim), dtype=torch.float32)
        cache_index = torch.full((batch,), action_horizon, dtype=torch.long)
        dummy_inputs.extend([cache, cache_index])

    input_names = [f"obs::{k}" for k in obs_keys] + [f"goal::{k}" for k in goal_keys]
    if is_recurrent:
        input_names += ["h0", "c0"]
    if getattr(wrapper, "requires_action_cache", False):
        input_names += ["cached_actions", "cache_index"]

    default_outputs = ["actions"] + (["h1", "c1"] if is_recurrent else [])
    output_names = getattr(wrapper, "output_names", default_outputs)

    dyn_axes = None  # fully-static shapes

    print("[*] Created dummy inputs with fully-static shapes")
    return dummy_inputs, input_names, output_names, dyn_axes


def export_to_onnx(wrapper, dummy_inputs, input_names, output_names, dyn_axes, onnx_path):
    """Export the wrapper network to a fully-static ONNX file."""
    import torch

    export_kwargs = dict(
        model=wrapper,
        args=tuple(dummy_inputs),
        f=onnx_path.as_posix(),
        opset_version=17,
        input_names=input_names,
        output_names=output_names,
        do_constant_folding=True,
    )
    if dyn_axes is not None:
        export_kwargs["dynamic_axes"] = dyn_axes

    print(f"[*] Exporting static-shape ONNX to {onnx_path}")
    torch.onnx.export(**export_kwargs)
    print("[*] ONNX export complete")


def main():
    args = parse_args()
    _apply_tracing_safe_overrides()
    if not (args.folder or (args.ckpt and args.config)):
        raise ValueError("Either --folder or both --ckpt and --config must be provided")

    config_path, ckpt_path, onnx_path = find_paths(args.folder) if args.folder else (
        pathlib.Path(args.config).expanduser(),
        pathlib.Path(args.ckpt).expanduser(),
        pathlib.Path(args.out if args.out else f"{pathlib.Path(args.ckpt).stem}.onnx").expanduser(),
    )

    config = load_config(config_path)
    policy, ckpt_dict = load_policy(ckpt_path=ckpt_path, device=args.device)
    print("[DEBUG] ckpt_dict keys:", list(ckpt_dict.keys()))
    print("[DEBUG] normalization-ish subkeys:",
          {k: list(ckpt_dict[k].keys()) for k in ckpt_dict if isinstance(ckpt_dict[k], dict) and "norm" in k.lower()})

    action_norm = extract_action_normalization(ckpt_dict)
    wrapper, obs_keys, goal_keys, is_recurrent = wrap_policy(policy, config, action_norm or {}, ckpt_dict)
    dummy_inputs, input_names, output_names, dyn_axes = create_dummy_inputs(wrapper, obs_keys, goal_keys, is_recurrent, config)
    export_to_onnx(wrapper, dummy_inputs, input_names, output_names, dyn_axes, onnx_path)


if __name__ == '__main__':
    main()

