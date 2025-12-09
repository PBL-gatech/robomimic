"""
scripts/run_PatcherBot_diffusionAgent.py

Diffusion-policy-only offline evaluator for PatcherBot datasets.
At each environment step, it computes and logs the full action sequence
predicted by the diffusion policy (length = action_horizon), instead of
only the single action consumed that step.

This script does NOT modify env_patcher.py. It uses the same dataset-backed
environment and frame-stacking setup as run_PatcherBot_agent.py, but calls the
underlying diffusion policy's trajectory method directly to capture the full
sequence and unnormalize it.
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, List

import h5py
import numpy as np

import robomimic.utils.file_utils as FileUtils
import robomimic.utils.torch_utils as TorchUtils
import robomimic.utils.obs_utils as ObsUtils
import robomimic.utils.python_utils as PyUtils

from robomimic.envs.env_patcher import create_env_patcher


def _safe_get(obj: Any, key: str, default: Any = None) -> Any:
    if obj is None:
        return default
    if isinstance(obj, Mapping):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _to_list(value: Optional[Iterable]) -> list:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def _resolve_checkpoints(agent_arg: str) -> Sequence[Path]:
    path_str = str(Path(agent_arg).expanduser())
    direct_path = Path(path_str)
    paths: List[Path] = []
    if direct_path.is_file():
        paths = [direct_path]
    elif direct_path.is_dir():
        paths = sorted(p for p in direct_path.glob("*.pth") if p.is_file())
    else:
        from glob import glob
        matched = sorted(Path(p) for p in glob(path_str))
        paths = [p for p in matched if p.is_file()]
    if not paths:
        raise FileNotFoundError(f"No checkpoints found for {agent_arg}")
    return paths


def _extract_obs_modalities(cfg) -> Dict[str, list]:
    obs_cfg = _safe_get(_safe_get(cfg.observation, "modalities"), "obs")
    if obs_cfg is None:
        return {}
    if hasattr(obs_cfg, "to_dict"):
        obs_dict = obs_cfg.to_dict()
    elif isinstance(obs_cfg, Mapping):
        obs_dict = dict(obs_cfg)
    else:
        obs_dict = {}
    modalities: Dict[str, list] = {}
    for modality, keys in obs_dict.items():
        modalities[str(modality)] = [str(k) for k in _to_list(keys)]
    return modalities


def _ensure_key_in_modalities(modalities: Dict[str, list], key: Optional[str], modality: str) -> None:
    if key is None:
        return
    entries = modalities.setdefault(modality, [])
    if key not in entries:
        entries.append(key)


def _infer_rgb_shape(processed_shape, rgb_encoder, rgb_rand_kwargs) -> Sequence[int]:
    if processed_shape:
        processed_shape = tuple(int(s) for s in processed_shape)
        if len(processed_shape) >= 3:
            c, h, w = processed_shape[:3]
            return (h, w, c)
        if len(processed_shape) == 2:
            c, h = processed_shape
            return (h, h, c)
    crop_h = _safe_get(rgb_rand_kwargs, "crop_height")
    crop_w = _safe_get(rgb_rand_kwargs, "crop_width")
    if crop_h is not None and crop_w is not None:
        height, width = int(crop_h), int(crop_w)
    else:
        height = int(_safe_get(rgb_encoder, "height", 84))
        width = int(_safe_get(rgb_encoder, "width", 84))
    channels = int(_safe_get(rgb_encoder, "channels", 3))
    return (height, width, channels)


def _infer_low_dim_shape(key: Optional[str]) -> Sequence[int]:
    if not key:
        return (1,)
    name = key.lower()
    if "pipette" in name:
        return (3,)
    if "stage" in name:
        return (3,)
    if "resist" in name:
        return (1,)
    return (1,)


def _infer_derived_spec(key: str, existing_specs: Mapping[str, Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    suffix_map = {
        "_velocities": "",
        "_velocity": "",
        "_vels": "",
        "_vel": "",
        "_deltas": "",
        "_delta": "",
        "_diff": "",
    }
    for suffix, replacement in suffix_map.items():
        if key.endswith(suffix):
            base = key[: -len(suffix)] + replacement
            if base in existing_specs:
                base_spec = existing_specs[base]
                return {
                    "shape": tuple(base_spec.get("shape", (1,))),
                    "dtype": np.float32,
                    "fill_value": 0.0,
                    "transform": "diff",
                    "source_key": base,
                    "modality": base_spec.get("modality", "low_dim"),
                }
    return None


def _build_obs_specs(cfg, ckpt_dict, obs_modalities: Dict[str, list]) -> Dict[str, Dict[str, Any]]:
    shape_meta = ckpt_dict.get("shape_metadata", {}) or {}
    obs_shapes = shape_meta.get("all_shapes", {}) or {}
    encoder_cfg = _safe_get(cfg.observation, "encoder")
    rgb_encoder = _safe_get(encoder_cfg, "rgb")
    rgb_rand_kwargs = _safe_get(rgb_encoder, "obs_randomizer_kwargs")
    specs: Dict[str, Dict[str, Any]] = {}
    for modality, keys in obs_modalities.items():
        for key in keys:
            processed_shape = obs_shapes.get(key)
            spec: Dict[str, Any] = {"modality": modality}
            if modality == "rgb":
                spec["dtype"] = np.uint8
                spec["fill_value"] = 0
                spec["shape"] = tuple(_infer_rgb_shape(processed_shape, rgb_encoder, rgb_rand_kwargs))
            else:
                spec["dtype"] = np.float32
                spec["fill_value"] = 0.0
                if processed_shape is not None:
                    spec["shape"] = tuple(int(s) for s in processed_shape)
                else:
                    spec["shape"] = tuple(_infer_low_dim_shape(key))
            specs[key] = spec
    for key, spec in list(specs.items()):
        if key not in obs_shapes:
            derived = _infer_derived_spec(key, specs)
            if derived:
                for derived_key, derived_value in derived.items():
                    spec.setdefault(derived_key, derived_value)
    return specs


def _stack_goal_if_needed(goal_dict, stack):
    if goal_dict is None or stack <= 1 or not isinstance(goal_dict, dict):
        return goal_dict
    stacked = {}
    for key, value in goal_dict.items():
        arr = np.asarray(value)
        if arr.ndim == 0:
            arr = np.repeat(arr[None], stack, axis=0)
        elif arr.shape[0] != stack:
            arr = np.repeat(arr[np.newaxis, ...], stack, axis=0)
        stacked[key] = arr
    return stacked


def _unnormalize_action_sequence(seq: np.ndarray, policy_wrapper) -> np.ndarray:
    """
    Unnormalize a sequence of actions using the same logic as RolloutPolicy.__call__.

    Args:
        seq: (T, Da) normalized actions in [-1, 1]
        policy_wrapper: RolloutPolicy instance with action_normalization_stats
    Returns:
        (T, Da) unnormalized actions
    """
    stats = getattr(policy_wrapper, "action_normalization_stats", None)
    if not stats:
        return seq
    action_keys = policy_wrapper.policy.global_config.train.action_keys
    action_shapes = {k: stats[k]["offset"].shape[1:] for k in stats}

    # treat T as batch dimension
    ac_dict = PyUtils.vector_to_action_dict(seq, action_shapes=action_shapes, action_keys=action_keys)
    ac_dict = ObsUtils.unnormalize_dict(ac_dict, normalization_stats=stats)

    # optional rot_6d conversion like RolloutPolicy
    action_config = policy_wrapper.policy.global_config.train.action_config
    for key, value in ac_dict.items():
        this_format = action_config[key].get("format", None)
        if this_format == "rot_6d":
            import torch
            rot_6d = torch.from_numpy(value)
            conversion_format = action_config[key].get("convert_at_runtime", "rot_axis_angle")
            if conversion_format == "rot_axis_angle":
                rot = TorchUtils.rot_6d_to_axis_angle(rot_6d=rot_6d).numpy()
            elif conversion_format == "rot_euler":
                rot = TorchUtils.rot_6d_to_euler_angles(rot_6d=rot_6d, convention="XYZ").numpy()
            else:
                raise ValueError
            ac_dict[key] = rot

    seq_unnorm = PyUtils.action_dict_to_vector(ac_dict, action_keys=action_keys)
    return seq_unnorm


def _append_vector(prefix: str, array: Optional[Iterable], record: Dict[str, Any]) -> None:
    if array is None:
        return
    arr = np.asarray(array, dtype=np.float32).reshape(-1)
    for idx, value in enumerate(arr):
        record[f"{prefix}_{idx}"] = float(value)


def main():
    df_path = r"C:\Users\sa-forest\Documents\GitHub\robomimic\df_patcherBot\PipetteFinding\v0_160\20251007110009"
    data_path = r"C:\Users\sa-forest\Documents\GitHub\holypipette-pbl\experiments\Datasets\PatcherBot_test_dataset_v0_160\PatcherBot_test_dataset_v0_160_find_pipette.hdf5"
    base_path = r"C:\Users\sa-forest\Documents\GitHub\robomimic\df_patcherBot\PipetteFinding\results"
    ap = argparse.ArgumentParser()
    ap.add_argument("--agent", default=df_path, help="Path to diffusion .pth checkpoint (or folder / glob)")
    ap.add_argument("--dataset", default=data_path, help="Path to PatcherBot .hdf5 dataset")
    ap.add_argument("--horizon", type=int, default=None, help="Max rollout steps (defaults to demo length)")
    ap.add_argument("--frame_stack", type=int, default=None, help="Frame stack override; defaults to policy config")
    ap.add_argument("--eps", type=float, default=-1.0, help="Success epsilon")
    ap.add_argument("--per_seq_csv", type=str, default=None, help="Optional path to save per-sequence metrics as CSV")
    ap.add_argument("--seq_elements_csv", type=str, default=r"C:\Users\sa-forest\Documents\GitHub\robomimic\df_patcherBot\PipetteFinding\results\v0_160\results_df_PatcherBot_traj_v0_160_0.csv", help="Optional path to save per-element sequence details as CSV")
    ap.add_argument("--rollout_metadata", type=str, default=r"C:\Users\sa-forest\Documents\GitHub\robomimic\df_patcherBot\PipetteFinding\results\v0_160\metadata_df_PatcherBot_traj_v0_160_0.json", help="Optional path to save rollout metadata as JSON")
    args = ap.parse_args()

    ckpt_paths = _resolve_checkpoints(args.agent)
    device = TorchUtils.get_torch_device(try_to_use_cuda=True)

    per_seq_csv_target = Path(args.per_seq_csv).expanduser() if args.per_seq_csv else None
    elem_csv_target = Path(args.seq_elements_csv).expanduser() if args.seq_elements_csv else None
    per_seq_rows: List[Dict[str, Any]] = []
    elem_rows: List[Dict[str, Any]] = []

    for idx, ckpt_path in enumerate(ckpt_paths, start=1):
        print(f"[INFO] evaluating checkpoint {ckpt_path} ({idx}/{len(ckpt_paths)})")
        policy, ckpt_dict = FileUtils.policy_from_checkpoint(ckpt_path=str(ckpt_path), device=device, verbose=True)
        cfg, _ = FileUtils.config_from_checkpoint(ckpt_path=str(ckpt_path), ckpt_dict=ckpt_dict)

        algo_name = str(ckpt_dict.get("algo_name", "")).lower()
        if algo_name != "diffusion_policy":
            raise ValueError(f"This script is diffusion-only. Loaded algo={algo_name}")
        print(f"[INFO] loaded algorithm: {algo_name}")

        # discover policy frame_stack
        policy_frame_stack = None
        policy_impl = getattr(policy, "policy", None)
        if policy_impl is not None:
            cfg_policy = getattr(policy_impl, "global_config", None)
            if cfg_policy is not None:
                try:
                    cfg_stack = getattr(cfg_policy.train, "frame_stack", None)
                    if cfg_stack is not None:
                        policy_frame_stack = int(cfg_stack)
                except AttributeError:
                    policy_frame_stack = None
        if policy_frame_stack is None:
            cfg_stack = getattr(cfg.train, "frame_stack", None)
            if cfg_stack is not None:
                policy_frame_stack = int(cfg_stack)
        if policy_frame_stack is not None and policy_frame_stack <= 0:
            policy_frame_stack = None
        requested_stack = args.frame_stack if (args.frame_stack is not None and args.frame_stack > 0) else None
        frame_stack = requested_stack or policy_frame_stack or 1
        frame_stack = max(int(frame_stack), 1)
        if requested_stack and policy_frame_stack and requested_stack != policy_frame_stack:
            print(f"[WARN] overriding policy frame_stack={policy_frame_stack} with requested value {requested_stack}")
        else:
            print(f"[INFO] using frame_stack={frame_stack} (policy default: {policy_frame_stack or 'n/a'})")

        # horizon
        horizon = args.horizon
        if horizon is None:
            with h5py.File(args.dataset, "r") as h5:
                demo_id = sorted(h5["data"].keys())[0]
                horizon = h5["data"][demo_id]["actions"].shape[0]
            args.horizon = horizon
            print(f"[INFO] setting horizon to {horizon} from demo length")

        # build obs keys & specs
        obs_modalities = _extract_obs_modalities(cfg)
        rgb_keys = obs_modalities.get("rgb", [])
        image_key = rgb_keys[0] if rgb_keys else "camera_image"
        low_dim_keys = obs_modalities.get("low_dim", [])
        pipette_key = next((k for k in low_dim_keys if "pipette" in k.lower()), "pipette_positions")
        stage_key = next((k for k in low_dim_keys if "stage" in k.lower()), "stage_positions")
        resistance_key = next((k for k in low_dim_keys if "resist" in k.lower()), "resistance")
        _ensure_key_in_modalities(obs_modalities, image_key, "rgb")
        _ensure_key_in_modalities(obs_modalities, pipette_key, "low_dim")
        _ensure_key_in_modalities(obs_modalities, stage_key, "low_dim")
        _ensure_key_in_modalities(obs_modalities, resistance_key, "low_dim")
        obs_key_specs = _build_obs_specs(cfg, ckpt_dict, obs_modalities)

        # create dataset-backed env
        env = create_env_patcher(
            dataset_path=args.dataset,
            frame_stack=frame_stack,
            success_epsilon=args.eps,
            horizon=horizon,
            image_key=image_key,
            pipette_key=pipette_key,
            stage_key=stage_key,
            resistance_key=resistance_key,
            obs_modalities=obs_modalities,
            obs_key_specs=obs_key_specs,
        )
        env_frame_stack = getattr(env, "frame_stack", frame_stack)

        # initialize policy
        policy.start_episode()
        obs = env.reset()
        state = env.get_state()
        obs = env.reset_to(state)
        goal = _stack_goal_if_needed(env.get_goal(), env_frame_stack)

        # diffusion horizons
        To = int(policy.policy.algo_config.horizon.observation_horizon)
        Ta = int(policy.policy.algo_config.horizon.action_horizon)

        def _prep(ob, goal):
            ob_t = policy._prepare_observation(ob, batched_ob=False)
            goal_t = policy._prepare_observation(goal, batched_ob=False) if goal is not None else None
            return ob_t, goal_t

        seq_counter = 0
        for _ in range(int(horizon)):
            t = env._t
            # prepare inputs for underlying diffusion policy
            ob_t, goal_t = _prep(obs, goal)

            # full sequence inference
            start_t = time.perf_counter()
            traj_torch = policy.policy._get_action_trajectory(obs_dict=ob_t, goal_dict=goal_t)
            latency_ms = (time.perf_counter() - start_t) * 1000.0

            # (1, Ta, Da) -> (Ta, Da), normalized; then unnormalize
            seq_norm = traj_torch.detach().cpu().numpy()[0]
            seq = _unnormalize_action_sequence(seq_norm, policy)

            # ground truth sequence from dataset
            gt_seq = env._actions_gt[t : min(t + Ta, env._actions_gt.shape[0])].astype(np.float32)
            if gt_seq.shape[0] < Ta:
                pad = np.repeat(gt_seq[-1:,...], Ta - gt_seq.shape[0], axis=0)
                gt_seq = np.concatenate([gt_seq, pad], axis=0)

            # elementwise error and summary
            diff = seq - gt_seq
            mse = float(np.mean(np.square(diff)))
            mae = float(np.mean(np.abs(diff)))
            rmse = float(np.sqrt(max(np.mean(np.square(diff)), 0.0)))

            per_seq_rows.append({
                "checkpoint": ckpt_path.name,
                "seq_index": int(seq_counter),
                "t_start": int(t),
                "latency_ms": float(latency_ms),
                "seq_len": int(Ta),
                "mse": mse,
                "mae": mae,
                "rmse": rmse,
            })

            # optional per-element logging
            if elem_csv_target is not None:
                for i in range(Ta):
                    row = {
                        "checkpoint": ckpt_path.name,
                        "seq_index": int(seq_counter),
                        "elem": int(i),
                        "t": int(t + i),
                    }
                    _append_vector("pred", seq[i], row)
                    _append_vector("gt", gt_seq[i], row)
                    _append_vector("err", diff[i], row)
                    elem_rows.append(row)

            # step env with first action from sequence
            act0 = seq[0].astype(np.float32).reshape(-1)
            obs, r, done, info = env.step(act0)
            goal = _stack_goal_if_needed(env.get_goal(), env_frame_stack)
            seq_counter += 1
            if done:
                break

        # write metadata (optional)
        if args.rollout_metadata:
            meta_path = Path(args.rollout_metadata).expanduser()
            meta_path.parent.mkdir(parents=True, exist_ok=True)
            env_horizon = getattr(env, "_horizon", None)
            demo_length = getattr(env, "_N", None)
            ckpt_summary = {k: ckpt_dict[k] for k in ("epoch", "iteration", "global_step", "train_step", "model_epoch") if k in ckpt_dict}
            train_cfg = getattr(cfg, "train", None)
            config_seed = _safe_get(train_cfg, "seed") if train_cfg is not None else None
            demo_key = getattr(env, "_demo_id", None)
            if isinstance(demo_key, bytes):
                demo_key = demo_key.decode("utf-8")
            metadata = {
                "agent_path": str(ckpt_path),
                "dataset_path": str(Path(args.dataset).expanduser()),
                "demo_key": demo_key,
                "frame_stack_used": frame_stack,
                "policy_frame_stack": policy_frame_stack,
                "requested_frame_stack": requested_stack,
                "env_frame_stack": getattr(env, "frame_stack", frame_stack),
                "horizon": int(horizon) if horizon is not None else None,
                "env_horizon": env_horizon,
                "demo_length": demo_length,
                "rollout_steps": int(seq_counter),
                "algo_name": algo_name,
                "device": str(device),
                "eps": float(args.eps) if args.eps is not None else None,
                "diffusion_horizons": {"obs": To, "action": Ta},
                "checkpoint_summary": ckpt_summary or None,
                "config_seed": config_seed,
            }
            with meta_path.open("w", encoding="utf-8") as f:
                json.dump(metadata, f, indent=2, sort_keys=True)
            print(f"[RESULT] wrote rollout metadata to {meta_path}")

        if hasattr(env, "close"):
            try:
                env.close()
            except Exception:
                pass

    # write CSVs after processing all checkpoints
    if per_seq_csv_target and per_seq_rows:
        per_seq_csv_target.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = []
        for record in per_seq_rows:
            for key in record:
                if key not in fieldnames:
                    fieldnames.append(key)
        with per_seq_csv_target.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for record in per_seq_rows:
                writer.writerow({key: record.get(key, "") for key in fieldnames})
        print(f"[RESULT] wrote per-sequence metrics to {per_seq_csv_target}")

    if elem_csv_target and elem_rows:
        elem_csv_target.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = []
        for record in elem_rows:
            for key in record:
                if key not in fieldnames:
                    fieldnames.append(key)
        with elem_csv_target.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for record in elem_rows:
                writer.writerow({key: record.get(key, "") for key in fieldnames})
        print(f"[RESULT] wrote per-element sequence details to {elem_csv_target}")


if __name__ == "__main__":
    main()

