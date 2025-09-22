#!/usr/bin/env python
"""
Export a Robomimic policy checkpoint to ONNX Runtime format.

Usage (example):
    # Auto-find config and latest model in folder:
    python export_to_onnx.py --folder bc_patcherBot_trained_models_HEK_v0_015/v0_015/20250506223704/

    # Or specify files manually (original approach):
    python export_to_onnx.py --ckpt model_epoch_485.pth --config config.json --out policy.onnx
"""
import argparse, torch, json, pathlib, glob, re
from collections import OrderedDict

from robomimic.utils.file_utils import policy_from_checkpoint
from robomimic.utils import obs_utils as ObsUtils
import robomimic.utils.tensor_utils as TensorUtils


# Default folder used when no CLI args are provided (pointing to repo root)
# DEFAULT_FOLDER = pathlib.Path(__file__).parent.parent.parent.parent / 'bc_patcherBot' / 'v0_029' / '20250831172330'
# DEFAULT_FOLDER = pathlib.Path(__file__).parent.parent.parent.parent / 'bc_patcherBot' / 'v0_035' / '20250917152411' 
# DEFAULT_FOLDER = pathlib.Path(__file__).parent.parent.parent.parent / 'bc_patcherBot' / 'v0_036' / '20250918011750'
# DEFAULT_FOLDER = pathlib.Path(__file__).parent.parent.parent.parent / 'bc_patcherBot' / 'v0_037' / '20250921164912'
DEFAULT_FOLDER = pathlib.Path(__file__).parent.parent.parent.parent / 'bc_patcherBot' / 'v0_038' / '20250921221249'


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", help="Path to .pth checkpoint")
    parser.add_argument("--config", help="Path to config.json")
    parser.add_argument("--out", help="Output ONNX file")
    parser.add_argument("--folder", help="Path to folder containing checkpoint and config", default=str(DEFAULT_FOLDER))
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
    print("[load_config] rnn settings:", cfg["algo"]["rnn"])
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

    offset_t = torch.from_numpy(offset).view(1, 1, dim)
    scale_t = torch.from_numpy(scale).view(1, 1, dim)

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


def wrap_policy(policy, config, action_norm_stats):
    """
    Build a thin wrapper around the loaded Robomimic policy so that it can be
    exported to ONNX.

    Returns
    -------
    wrapper : nn.Module  -- Forward pass ready for `torch.onnx.export`
    obs_keys : List[str] -- Order of observations expected by wrapper
    goal_keys : List[str] -- Goal observation keys (can be empty)
    is_recurrent : bool  -- Whether the policy carries RNN state
    """
    import torch.nn as nn

    policy_obj = policy[0] if isinstance(policy, tuple) else policy
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
    is_recurrent = bool(config["algo"]["rnn"]["enabled"])

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
            if (self.act_offset is None) or (self.act_scale is None):
                return actions
            adict = {"actions": actions}
            stats = {"actions": {"offset": self.act_offset, "scale": self.act_scale}}
            ObsUtils.unnormalize_dict(adict, stats)
            return adict["actions"]

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


def create_dummy_inputs(wrapper, obs_keys, goal_keys, is_recurrent, config):
    """
    Build fixed-size zero tensors for export. All dimensions-including batch
    and sequence length-are constant, so OpenCV DNN will load the file.
    """
    import torch

    batch = 1
    seq = 1

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

    input_names = [f"obs::{k}" for k in obs_keys] + [f"goal::{k}" for k in goal_keys]
    if is_recurrent:
        input_names += ["h0", "c0"]

    output_names = ["actions"] + (["h1", "c1"] if is_recurrent else [])

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
    wrapper, obs_keys, goal_keys, is_recurrent = wrap_policy(policy, config, action_norm or {})
    dummy_inputs, input_names, output_names, dyn_axes = create_dummy_inputs(wrapper, obs_keys, goal_keys, is_recurrent, config)
    export_to_onnx(wrapper, dummy_inputs, input_names, output_names, dyn_axes, onnx_path)


if __name__ == '__main__':
    main()
