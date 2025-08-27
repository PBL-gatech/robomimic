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


# Default folder used when no CLI args are provided (pointing to repo root)

DEFAULT_FOLDER = pathlib.Path(__file__).parent.parent.parent.parent /'training'/ 'bc_patcherBot_trained_models_HEK_v0_021' / 'v0_021' / '20250527191420'
# C:\Users\sa-forest\Documents\GitHub\robomimic\training\bc_patcherBot_trained_models_HEK_v0_021\v0_021\20250527191420
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
    policy, dict = policy_from_checkpoint(
        ckpt_path = ckpt_path,
        device=torch.device(device),
        verbose=False
    )
    
    print("[*] Policy loaded")
    return policy


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
            self.rnn          = getattr(actor_net, "rnn", None)
            self.rnn_type     = getattr(self.rnn, "rnn_type", None)

        def forward(self, *tensors):
            # Split the incoming *args* into observation tensors and, if
            # recurrent, hidden state tensors
            num_obs    = len(self.obs_keys)
            obs_tensors = tensors[:num_obs]
            obs_dict    = {k: v for k, v in zip(self.obs_keys, obs_tensors)}

            if self.is_recurrent:
                if self.rnn_type == "GRU":
                    h0 = tensors[num_obs]
                    actions, h1 = self.actor_net.forward_step(obs_dict, rnn_state=h0)
                    if actions.ndimension() == 3:
                        actions = actions[:, 0]
                    return actions, h1
                else:  # default to LSTM behaviour
                    h0, c0 = tensors[num_obs], tensors[num_obs + 1]
                    actions, (h1, c1) = self.actor_net.forward_step(obs_dict, rnn_state=(h0, c0))
                    if actions.ndimension() == 3:
                        actions = actions[:, 0]
                    return actions, h1, c1
            else:
                actions = self.actor_net(obs_dict)
                # Some robomimic nets return a dict; normalise here
                if isinstance(actions, dict):
                    actions = actions.get("actions", next(iter(actions.values())))
                if actions.ndimension() == 3:
                    actions = actions[:, 0]
                return actions


    wrapper = OnnxPolicy(actor_net, obs_keys, is_recurrent).cpu().eval()
    print("[wrap_policy] Wrapper ready → obs:", obs_keys,
          "| recurrent:", is_recurrent)
    return wrapper, obs_keys, is_recurrent

def create_dummy_inputs(wrapper, obs_keys, is_recurrent, config):
    """
    Build fixed-size zero tensors for export. All dimensions are constant so
    the exported ONNX model expects a single observation step.
    """
    import torch

    batch = 1  # fixed

    dummy_inputs = [
        torch.zeros((batch, *wrapper.obs_shapes[k]), dtype=torch.float32)
        for k in obs_keys
    ]

    input_names = list(obs_keys)
    output_names = ['actions']

    if is_recurrent:
        num_layers  = config["algo"]["rnn"]["num_layers"]
        hidden_size = config["algo"]["rnn"]["hidden_dim"]
        rnn_type = getattr(wrapper.rnn, "rnn_type", "LSTM")

        dummy_inputs.append(torch.zeros((num_layers, batch, hidden_size), dtype=torch.float32))  # h0
        input_names.append('h0')
        output_names.append('h1')

        if rnn_type != "GRU":
            dummy_inputs.append(torch.zeros((num_layers, batch, hidden_size), dtype=torch.float32))  # c0
            input_names.append('c0')
            output_names.append('c1')

    dyn_axes = None  # <--- turn off dynamic axes completely

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
        policy = load_policy(ckpt_path=ckpt_path, device=args.device)
        # inspect_policy(ckpt_path=ckpt_path, device=args.device)  # Uncomment to inspect policy and checkpoint
        wrapper, obs_keys, is_recurrent = wrap_policy(policy, config)
        dummy_inputs, input_names, output_names, dyn_axes = create_dummy_inputs(wrapper, obs_keys, is_recurrent, config)
        export_to_onnx(wrapper, dummy_inputs, input_names, output_names, dyn_axes, onnx_path)
    else:
        raise ValueError("Either --folder or both --ckpt and --config must be provided")

if __name__ == '__main__':
    main()
