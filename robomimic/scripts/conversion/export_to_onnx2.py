#!/usr/bin/env python
"""Export a Robomimic policy checkpoint to ONNX along with sidecar metadata."""
from __future__ import annotations

import argparse
import pathlib
from typing import Any, Mapping

from robomimic.utils import export_utils as ExportUtils


def _resolve_paths(args: argparse.Namespace) -> tuple[pathlib.Path, pathlib.Path, pathlib.Path]:
    if args.folder:
        config_path, ckpt_path, default_onnx = ExportUtils.find_paths(args.folder)
        if args.config:
            config_path = pathlib.Path(args.config).expanduser()
        if args.ckpt:
            ckpt_path = pathlib.Path(args.ckpt).expanduser()
        onnx_path = pathlib.Path(args.out).expanduser() if args.out else default_onnx
        return config_path, ckpt_path, onnx_path

    if not (args.ckpt and args.config):
        raise ValueError("Provide --folder or both --ckpt and --config")

    ckpt_path = pathlib.Path(args.ckpt).expanduser()
    config_path = pathlib.Path(args.config).expanduser()
    onnx_name = args.out if args.out else f"{ckpt_path.stem}.onnx"
    onnx_path = pathlib.Path(onnx_name).expanduser()
    return config_path, ckpt_path, onnx_path


def _algo_name(ckpt_dict: Mapping[str, Any]) -> str:
    return str(ckpt_dict.get("algo_name", "")).lower()


def main() -> None:
    # bc_path = r"C:\Users\sa-forest\Documents\GitHub\robomimic\bc_patcherBot\PipetteFinding\v0_142\20251004111420"
    # bc_path = r"C:\Users\sa-forest\Documents\GitHub\robomimic\bc_patcherBot\PipetteFinding\v0_140\20251003184552"
    # bc_path = r"C:\Users\sa-forest\Documents\GitHub\robomimic\bc_patcherBot\PipetteFinding\v0_120\20251002204916"
    # bc_path = r"C:\Users\sa-forest\Documents\GitHub\robomimic\bc_patcherBot\PipetteFinding\v0_150\20251006125401"
    # bc_path = r"C:\Users\sa-forest\Documents\GitHub\robomimic\bc_patcherBot\PipetteFinding\v0_160\20251006142923"
    # bc_path = r"C:\Users\sa-forest\Documents\GitHub\robomimic\bc_patcherBot\PipetteFinding\v0_170\20251006151641"
    # bc_path = r"C:\Users\sa-forest\Documents\GitHub\robomimic\bc_patcherBot\PipetteFinding\v0_191\20251008203200"
    # bc_path = r"C:\Users\sa-forest\Documents\GitHub\robomimic\bc_patcherBot\PipetteFinding\v0_190\20251008212647"
    # bc_path = r"C:\Users\sa-forest\Documents\GitHub\robomimic\bc_patcherBot\PipetteFinding\v0_200\20251009011619"
    # bc_path = r"C:\Users\sa-forest\Documents\GitHub\robomimic\bc_patcherBot\PipetteFinding\v0_201\20251009161156"
    # bc_path = r"C:\Users\sa-forest\Documents\GitHub\robomimic\bc_patcherBot\PipetteFinding\v0_300\20251009220645"
    # bc_path = r"C:\Users\sa-forest\Documents\GitHub\robomimic\bc_patcherBot\PipetteFinding\v0_400\20251010191724"
    # bc_path = r"C:\Users\sa-forest\Documents\GitHub\robomimic\bc_patcherBot\PipetteFinding\v0_430\20251012104524"
    # bc_path = r"C:\Users\sa-forest\Documents\GitHub\robomimic\bc_patcherBot\PipetteFinding\v0_432\20251014192119"
    # bc_path = r"C:\Users\sa-forest\Documents\GitHub\robomimic\bc_patcherBot\PipetteFinding\v0_435\20251017184634"
    # bc_path = r"C:\Users\sa-forest\Documents\GitHub\robomimic\bc_patcherBot\PipetteFinding\v0_501\20251019132555"
    bc_path = r"C:\Users\sa-forest\Documents\GitHub\robomimic\bc_patcherBot\PipetteFinding\v0_510\20251020194550"

    parser = ExportUtils.make_export_arg_parser(default_folder=bc_path)
    args = parser.parse_args()

    config_path, ckpt_path, onnx_path = _resolve_paths(args)

    export_dir = onnx_path.parent / onnx_path.stem
    export_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = export_dir / (onnx_path.name if onnx_path.suffix else f"{onnx_path.name}.onnx")

    config = ExportUtils.load_config(config_path)
    policy, ckpt_dict = ExportUtils.load_policy(ckpt_path, device=args.device)

    if _algo_name(ckpt_dict) == "diffusion_policy":
        raise NotImplementedError("This exporter only handles standard BC-style checkpoints")

    rollout_policy = policy[0] if isinstance(policy, tuple) else policy

    wrapper, obs_keys, goal_keys, is_recurrent, obs_shapes, goal_shapes, rnn_cfg = ExportUtils.build_actor_export_module(
        rollout_policy,
        config,
    )

    obs_norm = ExportUtils.extract_obs_normalization(rollout_policy, obs_keys, goal_keys)
    action_norm = ExportUtils.extract_action_normalization(ckpt_dict)
    if hasattr(wrapper, 'set_normalization_stats'):
        wrapper.set_normalization_stats(obs_norm, action_norm)

    dummy_inputs, input_names, output_names = ExportUtils.create_dummy_inputs(
        obs_shapes,
        goal_shapes,
        is_recurrent=is_recurrent,
        rnn_cfg=rnn_cfg,
    )

    ExportUtils.export_to_onnx(wrapper, dummy_inputs, input_names, output_names, onnx_path)

    obs_stats_path = None
    if obs_norm:
        candidate = export_dir / "observation_normalization.npz"
        if ExportUtils.save_observation_normalization_npz(obs_norm, candidate):
            obs_stats_path = candidate

    action_stats_path = None
    candidate_action = export_dir / "action_normalization.npz"
    if ExportUtils.save_action_normalization_npz(action_norm, candidate_action):
        action_stats_path = candidate_action

    manifest_path = export_dir / "model.json"
    metadata = ExportUtils.make_metadata(
        ckpt_path=ckpt_path,
        onnx_path=onnx_path,
        input_names=input_names,
        output_names=output_names,
        obs_keys=obs_keys,
        goal_keys=goal_keys,
        is_recurrent=is_recurrent,
        obs_stats_filename=obs_stats_path.name if obs_stats_path else None,
        action_stats_filename=action_stats_path.name if action_stats_path else None,
        needs_postprocessing=False,
        export_directory=export_dir,
        requires_preprocessing=False,
        requires_postprocessing=False,
    )
    ExportUtils.write_metadata(metadata, manifest_path)

    print("[*] Export complete:")
    print(f"    Export folder: {export_dir}")
    print(f"    ONNX model: {onnx_path}")
    if obs_stats_path:
        print(f"    Observation stats: {obs_stats_path}")
    if action_stats_path:
        print(f"    Action stats: {action_stats_path}")
    print(f"    Manifest: {manifest_path}")
    print(f"    Observations: {obs_keys}")
    print(f"    Goals: {goal_keys if goal_keys else 'None'}")
    print(f"    Recurrent: {is_recurrent}")


if __name__ == "__main__":
    main()




