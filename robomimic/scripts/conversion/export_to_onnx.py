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
from robomimic.utils.file_utils import policy_from_checkpoint
from robomimic.utils import obs_utils as ObsUtils


# Default folder used when no CLI args are provided (pointing to repo root)

DEFAULT_FOLDER = pathlib.Path(__file__).parent.parent.parent.parent / 'training' /'bc_patcherBot_trained_models_HEK_v0_024' / 'v0_024' / '20250603001409'


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
        ckpt_path = ckpt_path,
        device=torch.device(device),
        verbose=False
    )
    
    print("[*] Policy loaded")
    return policy, ckpt_dict


def inspect_policy(ckpt_path, device):
    """
    Load raw checkpoint and policy, then print their top-level keys/attributes
    for inspection.
    """
    print(f"[*] Inspecting checkpoint: {ckpt_path}")
    raw_ckpt = torch.load(ckpt_path, map_location=device)
    print("[*] Raw checkpoint keys:", list(raw_ckpt.keys()))
    if "model" in raw_ckpt:
        print("[*]   → model sub-dict keys:", list(raw_ckpt["model"].keys()))

    policy, ckpt_dict = policy_from_checkpoint(
        ckpt_path=ckpt_path,
        device=torch.device(device),
        verbose=False
    )
    print("[*] Extracted ckpt_dict top-level keys:", list(ckpt_dict.keys()))
    print("[*] Policy public attributes:", [k for k in dir(policy) if not k.startswith("_")])
    return policy, ckpt_dict


def load_config(config_path):
    """
    Load JSON config and print relevant sections.
    Returns the parsed dict.
    """
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
    Accepts offset/scale or mean/std.
    """
    # common locations across forks
    candidates = [
        ("normalization_stats", "actions"),
        ("action_normalization_stats", None),
        ("stats", "actions"),
        ("normalization", "actions"),
    ]

    stats = None
    for top, sub in candidates:
        node = ckpt_dict.get(top)
        if node is None:
            continue
        if sub is None and isinstance(node, dict):
            stats = node
        elif sub is not None and isinstance(node, dict):
            stats = node.get(sub)
        if stats is not None:
            break

    # rare: directly at top level
    if stats is None and isinstance(ckpt_dict, dict):
        if all(k in ckpt_dict for k in ("offset", "scale")) or all(k in ckpt_dict for k in ("mean", "std")):
            stats = ckpt_dict

    if stats is None or not isinstance(stats, dict):
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
    scale  = _to_1d_np(scale)
    D = int(offset.shape[0])

    import numpy as np
    offset_t = torch.from_numpy(offset).view(1, 1, D)
    scale_t  = torch.from_numpy(scale).view(1, 1, D)

    return {"actions": {"offset": offset_t, "scale": scale_t}}


def wrap_policy(policy, config):
    """
    Build a thin wrapper around the loaded Robomimic policy so that it can be
    exported to ONNX.  

    Returns
    -------
    wrapper : nn.Module  -- Forward pass ready for `torch.onnx.export`
    obs_keys : List[str] -- Order of observations expected by wrapper
    is_recurrent : bool  -- Whether the policy carries RNN state
    """
    import torch.nn as nn

    policy_obj = policy[0] if isinstance(policy, tuple) else policy
    inner      = policy_obj.policy                     # nn.Module

    # choose the module that really implements forward
    if hasattr(inner, "nets"):
        nets = inner.nets

        # `ModuleDict` supports ["key"] but not .get()
        if isinstance(nets, (dict, torch.nn.ModuleDict)):
            if "policy" in nets:        # Robomimic default
                actor_net = nets["policy"]
            elif "actor" in nets:
                actor_net = nets["actor"]
            else:                       # fallback to first entry
                actor_net = next(iter(nets.values()))
    else:
        actor_net = inner

    if hasattr(actor_net, "obs_shapes"):
        obs_shapes = actor_net.obs_shapes
    elif hasattr(inner, "obs_shapes"):
        obs_shapes = inner.obs_shapes
    else:
        raise AttributeError("Could not locate obs_shapes on policy. Check your Robomimic version.")

    obs_keys = list(obs_shapes.keys())               # order of observations expected by wrapper    

    # use config to determine recurrence
    is_recurrent = bool(config["algo"]["rnn"]["enabled"])


    class OnnxPolicy(nn.Module):
        def __init__(self, actor_net, obs_keys, is_recurrent):
            super().__init__()
            self.actor_net    = actor_net
            self.obs_keys     = obs_keys
            self.is_recurrent = is_recurrent
            # expose these so create_dummy_inputs() can see them:
            self.obs_shapes   = obs_shapes
            self.rnn        = getattr(actor_net, "rnn", None)

            # Action unnormalization parameters (registered as buffers → baked into ONNX)
            # We fetch them from the outer scope via closure at construction time.
            self.act_offset = None
            self.act_scale  = None

        @staticmethod
        def _extract_actions(actions):
            if isinstance(actions, dict):
                return actions.get("actions", next(iter(actions.values())))
            return actions

        def _unnormalize_actions(self, actions):
            # Use robomimic's own utility for inverse normalization when buffers are present
            if (self.act_offset is None) or (self.act_scale is None):
                return actions
            adict = {"actions": actions}
            stats = {"actions": {"offset": self.act_offset, "scale": self.act_scale}}
            ObsUtils.unnormalize_dict(adict, stats)  # in-place
            return adict["actions"]

        def forward(self, *tensors):
            # Split the incoming *args* into observation tensors and, if
            # recurrent, hidden state (h0, c0)
            obs_tensors = tensors[:len(self.obs_keys)]
            obs_dict    = {k: v for k, v in zip(self.obs_keys, obs_tensors)}

            if self.is_recurrent:
                h0, c0 = tensors[-2], tensors[-1]
                actions, (h1, c1) = self.actor_net(obs_dict,
                                                   rnn_init_state=(h0, c0),
                                                   return_state=True)
                actions = self._extract_actions(actions)
                actions = self._unnormalize_actions(actions)
                return actions, h1, c1
            else:
                actions = self.actor_net(obs_dict)
                # Some robomimic nets return a dict; normalise here
                actions = self._extract_actions(actions)
                actions = self._unnormalize_actions(actions)
                return actions


    # Build wrapper first, then attach buffers from extracted normalization
    wrapper = OnnxPolicy(actor_net, obs_keys, is_recurrent).cpu().eval()
    print("[wrap_policy] Wrapper ready → obs:", obs_keys,
          "| recurrent:", is_recurrent)
    return wrapper, obs_keys, is_recurrent


def create_dummy_inputs(wrapper, obs_keys, is_recurrent, config):
    """
    Build fixed-size zero tensors for export.  All dimensions—including batch
    and sequence length—are constant, so OpenCV DNN will load the file.
    """
    import torch

    batch = 1                                           # fixed
    seq   = int(config["train"].get("seq_length", 1))   # fixed

    dummy_inputs = [
        torch.zeros((batch, seq, *wrapper.obs_shapes[k]), dtype=torch.float32)
        for k in obs_keys
    ]

    if is_recurrent:
        num_layers  = config["algo"]["rnn"]["num_layers"]
        hidden_size = config["algo"]["rnn"]["hidden_dim"]
        dummy_inputs += [
            torch.zeros((num_layers, batch, hidden_size), dtype=torch.float32),  # h0
            torch.zeros((num_layers, batch, hidden_size), dtype=torch.float32)   # c0
        ]

    input_names  = obs_keys + (['h0', 'c0'] if is_recurrent else [])
    output_names = ['actions'] + (['h1', 'c1'] if is_recurrent else [])

    dyn_axes = None          # <--- turn off dynamic axes completely

    print("[*] Created dummy inputs with fully-static shapes")
    return dummy_inputs, input_names, output_names, dyn_axes


def export_to_onnx(wrapper, dummy_inputs, input_names, output_names, dyn_axes, onnx_path):
    """
    Export the wrapper network to a fully-static ONNX file.
    """
    import torch

    export_kwargs = dict(
        model             = wrapper,
        args              = tuple(dummy_inputs),
        f                 = onnx_path.as_posix(),
        opset_version     = 17,
        input_names       = input_names,
        output_names      = output_names,
        do_constant_folding = True,
    )
    # Only pass dynamic_axes if they are defined
    if dyn_axes is not None:
        export_kwargs["dynamic_axes"] = dyn_axes

    print(f"[*] Exporting static-shape ONNX to {onnx_path}")
    torch.onnx.export(**export_kwargs)
    print("[✓] ONNX export complete")

def main():
    args = parse_args()
    if args.folder or (args.ckpt and args.config):
        config_path, ckpt_path, onnx_path = find_paths(args.folder) if args.folder else (
            pathlib.Path(args.config).expanduser(),
            pathlib.Path(args.ckpt).expanduser(),
            pathlib.Path(args.out if args.out else f"{pathlib.Path(args.ckpt).stem}.onnx").expanduser()
        )
        # load config early
        config = load_config(config_path)
        policy, ckpt_dict = load_policy(ckpt_path=ckpt_path, device=args.device)
        print("[DEBUG] ckpt_dict keys:", list(ckpt_dict.keys()))
        print("[DEBUG] normalization-ish subkeys:",
             {k: list(ckpt_dict[k].keys()) for k in ckpt_dict if isinstance(ckpt_dict[k], dict) and "norm" in k.lower()})
        # Extract action normalization (from checkpoint) and attach to wrapper as buffers
        action_norm = extract_action_normalization(ckpt_dict)
        wrapper, obs_keys, is_recurrent = wrap_policy(policy, config)
        # If we found normalization stats, register buffers on the wrapper so they are baked into ONNX
        if action_norm and 'actions' in action_norm:
            wrapper.register_buffer('act_offset', action_norm['actions']['offset'].to(torch.float32))
            wrapper.register_buffer('act_scale',  action_norm['actions']['scale'].to(torch.float32))
            print('[*] Registered action unnormalization buffers on wrapper')
        else:
            print('[*] No action normalization found; ONNX will output normalized actions ([-1, 1])')
        dummy_inputs, input_names, output_names, dyn_axes = create_dummy_inputs(wrapper, obs_keys, is_recurrent, config)
        export_to_onnx(wrapper, dummy_inputs, input_names, output_names, dyn_axes, onnx_path)
    else:
        raise ValueError("Either --folder or both --ckpt and --config must be provided")

if __name__ == '__main__':
    main()
