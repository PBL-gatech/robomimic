"""Shared PatcherBot evaluation utilities used by scripts and GUI wrappers."""

from __future__ import annotations

import csv
import json
import statistics
import time
from collections import defaultdict
from datetime import datetime
from glob import glob
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

import h5py
import numpy as np

CSV_PREFIX = "results_bc_PatcherBot_"
METADATA_PREFIX = "metadata_bc_PatcherBot_"


def _safe_get(obj: Any, key: str, default: Any = None) -> Any:
    if obj is None:
        return default
    if isinstance(obj, Mapping):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _to_list(value: Optional[Iterable[Any]]) -> List[Any]:
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
        matched = sorted(Path(p) for p in glob(path_str))
        paths = [p for p in matched if p.is_file()]
    if not paths:
        raise FileNotFoundError(f"No checkpoints found for {agent_arg}")
    return paths


def sanitize_checkpoint_name(path_or_name: str) -> str:
    return Path(path_or_name).name.replace("/", "_").replace("\\", "_")


def sanitize_demo_name(demo_key: Optional[str]) -> str:
    if demo_key is None:
        return "demo"
    return str(demo_key).replace("/", "_").replace("\\", "_")


def get_checkpoint_csv_path(checkpoint_path: str, csv_dir: Path, demo_key: Optional[str] = None) -> Path:
    stem = f"{CSV_PREFIX}{sanitize_checkpoint_name(checkpoint_path)}"
    if demo_key is not None:
        stem = f"{stem}_{sanitize_demo_name(demo_key)}"
    return csv_dir / f"{stem}.csv"


def get_checkpoint_metadata_path(checkpoint_path: str, metadata_dir: Path, demo_key: Optional[str] = None) -> Path:
    stem = f"{METADATA_PREFIX}{sanitize_checkpoint_name(checkpoint_path)}"
    if demo_key is not None:
        stem = f"{stem}_{sanitize_demo_name(demo_key)}"
    return metadata_dir / f"{stem}.json"


def list_dataset_demo_keys(dataset_path: Path) -> List[str]:
    with h5py.File(dataset_path, "r") as h5:
        keys = list(h5["data"].keys())
    decoded = [key.decode("utf-8") if isinstance(key, bytes) else str(key) for key in keys]
    return sorted(decoded)


def _extract_obs_modalities(cfg: Any) -> Dict[str, List[str]]:
    obs_cfg = _safe_get(_safe_get(cfg.observation, "modalities"), "obs")
    if obs_cfg is None:
        return {}
    if hasattr(obs_cfg, "to_dict"):
        obs_dict = obs_cfg.to_dict()
    elif isinstance(obs_cfg, Mapping):
        obs_dict = dict(obs_cfg)
    else:
        obs_dict = {}
    modalities: Dict[str, List[str]] = {}
    for modality, keys in obs_dict.items():
        modalities[str(modality)] = [str(k) for k in _to_list(keys)]
    return modalities


def _ensure_key_in_modalities(modalities: Dict[str, List[str]], key: Optional[str], modality: str) -> None:
    if key is None:
        return
    entries = modalities.setdefault(modality, [])
    if key not in entries:
        entries.append(key)


def _infer_rgb_shape(processed_shape: Optional[Iterable[int]], rgb_encoder: Any, rgb_rand_kwargs: Any) -> Sequence[int]:
    if processed_shape:
        processed_shape = tuple(int(s) for s in processed_shape)
        if len(processed_shape) >= 3:
            channels, height, width = processed_shape[:3]
            return (height, width, channels)
        if len(processed_shape) == 2:
            channels, height = processed_shape
            return (height, height, channels)
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
    if "pipette" in name or "stage" in name:
        return (3,)
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


def _build_obs_specs(cfg: Any, ckpt_dict: Mapping[str, Any], obs_modalities: Dict[str, List[str]]) -> Dict[str, Dict[str, Any]]:
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


def _extract_policy_action(output: Any) -> np.ndarray:
    if isinstance(output, np.ndarray):
        arr = output
    elif isinstance(output, (list, tuple)):
        arr = np.asarray(output)
    elif hasattr(output, "detach") and hasattr(output, "cpu"):
        arr = output.detach().cpu().numpy()
    elif isinstance(output, Mapping):
        for key in ("actions", "action", "pred_actions", "ac"):
            if key in output:
                return _extract_policy_action(output[key])
        first_val = next(iter(output.values()), None)
        if first_val is not None:
            return _extract_policy_action(first_val)
        arr = np.array([], dtype=np.float32)
    else:
        arr = np.asarray(output)
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim == 0:
        return arr.reshape(1)
    if arr.ndim > 1:
        arr = arr.reshape(arr.shape[0], -1)
        arr = arr[0]
    return arr.reshape(-1)


def _append_vector(prefix: str, array: Optional[Iterable[Any]], record: Dict[str, Any]) -> None:
    if array is None:
        return
    arr = np.asarray(array, dtype=np.float32).reshape(-1)
    for idx, value in enumerate(arr):
        record[f"{prefix}_{idx}"] = float(value)


def _build_step_record(
    step_idx: int,
    act_pred: np.ndarray,
    act_gt: np.ndarray,
    info: Mapping[str, Any],
    reward: float,
    latency_ms: Optional[float] = None,
) -> Dict[str, Any]:
    record: Dict[str, Any] = {
        "step": int(step_idx),
        "error_l2": float(info.get("error_l2", np.nan)),
    }
    t_val = info.get("t")
    if t_val is not None:
        record["env_t"] = int(t_val)
    record["reward"] = float(reward)
    _append_vector("gt", act_gt, record)
    _append_vector("pred", act_pred, record)
    err_vec = info.get("error_vec")
    _append_vector("err", err_vec, record)
    if err_vec is not None:
        _append_vector("abs_err", np.abs(err_vec), record)
    if latency_ms is not None:
        record["latency_ms"] = float(latency_ms)
    return record


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.generic,)):
        return value.item()
    return value


def _compute_loss_components(
    algo_name: str,
    loss_cfg: Optional[Mapping[str, Any]],
    act_pred: np.ndarray,
    act_gt: np.ndarray,
) -> Dict[str, float]:
    metrics: Dict[str, float] = {}
    diff = act_pred - act_gt
    mse = float(np.mean(np.square(diff)))
    mae = float(np.mean(np.abs(diff)))
    metrics["mse"] = mse
    metrics["mae"] = mae
    algo_name = (algo_name or "").lower()
    total = 0.0
    if loss_cfg is not None:
        l2_w = float(loss_cfg.get("l2_weight", 0.0))
        l1_w = float(loss_cfg.get("l1_weight", 0.0))
        cos_w = float(loss_cfg.get("cos_weight", 0.0))
        if l2_w:
            metrics["weighted_l2"] = l2_w * mse
            total += metrics["weighted_l2"]
        if l1_w:
            metrics["weighted_l1"] = l1_w * mae
            total += metrics["weighted_l1"]
        if cos_w:
            denom = float(np.linalg.norm(act_pred) * np.linalg.norm(act_gt))
            if denom > 1e-6:
                cos_sim = float(np.dot(act_pred, act_gt) / denom)
                metrics["cosine_similarity"] = cos_sim
                metrics["weighted_cos"] = cos_w * (1.0 - cos_sim)
                total += metrics["weighted_cos"]
    if total:
        metrics["weighted_total"] = total
    metrics.setdefault("rmse", float(np.sqrt(max(metrics["mse"], 0.0))))
    if "diffusion" in algo_name:
        metrics.setdefault("mae", mae)
    return metrics


def _summarize_losses(loss_totals: Dict[str, List[float]]) -> str:
    parts = []
    for name, (total, count) in sorted(loss_totals.items()):
        if count:
            parts.append(f"{name}={total / count:.6f}")
    return " | ".join(parts)


def _resolve_output_dir(config: Any, attr_name: str, fallback: Optional[Path] = None) -> Path:
    value = _safe_get(config, attr_name)
    if value:
        return Path(value).expanduser()
    if fallback is not None:
        return fallback
    raise ValueError(f"Missing required output directory: {attr_name}")


def _compute_dataset_horizon(dataset_path: Path) -> int:
    with h5py.File(dataset_path, "r") as h5:
        demo_id = sorted(h5["data"].keys())[0]
        return int(h5["data"][demo_id]["actions"].shape[0])


def _stack_goal_if_needed(goal_dict: Any, stack: int) -> Any:
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


def _compute_positional_trajectory_metrics(cfg: Any, acts: List[np.ndarray], env: Any) -> Optional[Dict[str, float]]:
    if not acts:
        return None
    try:
        seq_len = int(getattr(cfg.train, "seq_length", 16))
    except Exception:
        seq_len = 16

    action_array = np.asarray(acts).reshape(len(acts), -1)
    if action_array.shape[1] < 6 or not hasattr(env, "_pipette_positions"):
        return None

    pred_deltas = action_array[:, 3:6]
    start = max(0, seq_len - 1)
    end = min(start + pred_deltas.shape[0], len(env._pipette_positions))
    init_pos = env._pipette_positions[start]
    obs_pos = env._pipette_positions[start:end]
    n = min(pred_deltas.shape[0], obs_pos.shape[0])
    pred_deltas = pred_deltas[:n]
    obs_pos = obs_pos[:n]
    if n == 0:
        return None

    pred_pos = [init_pos]
    for delta in pred_deltas:
        pred_pos.append(pred_pos[-1] + delta)
    pred_pos_array = np.stack(pred_pos[1:])
    err = np.linalg.norm(pred_pos_array - obs_pos, axis=1)
    return {
        "steps": int(len(err)),
        "mean": float(err.mean()),
        "median": float(np.median(err)),
        "p95": float(np.percentile(err, 95)),
    }


def _write_step_records_csv(csv_path: Path, per_step_records: List[Dict[str, Any]]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: List[str] = []
    for record in per_step_records:
        for key in record:
            if key not in fieldnames:
                fieldnames.append(key)
    if "checkpoint" in fieldnames:
        fieldnames = ["checkpoint"] + [key for key in fieldnames if key != "checkpoint"]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in per_step_records:
            writer.writerow({key: record.get(key, "") for key in fieldnames})


def _load_checkpoint_context(checkpoint_path: str) -> Dict[str, Any]:
    import robomimic.utils.file_utils as FileUtils
    import robomimic.utils.torch_utils as TorchUtils

    ckpt_path = Path(checkpoint_path).expanduser()
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    device = TorchUtils.get_torch_device(try_to_use_cuda=True)
    policy, ckpt_dict = FileUtils.policy_from_checkpoint(ckpt_path=str(ckpt_path), device=device, verbose=True)
    cfg, _ = FileUtils.config_from_checkpoint(ckpt_path=str(ckpt_path), ckpt_dict=ckpt_dict)

    algo_name = str(ckpt_dict.get("algo_name", "")).lower()
    loss_cfg = None
    loss_section = _safe_get(cfg.algo, "loss")
    if loss_section is not None:
        if hasattr(loss_section, "to_dict"):
            loss_cfg = loss_section.to_dict()
        elif isinstance(loss_section, Mapping):
            loss_cfg = dict(loss_section)

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

    return {
        "checkpoint_path": ckpt_path,
        "device": device,
        "policy": policy,
        "ckpt_dict": ckpt_dict,
        "cfg": cfg,
        "algo_name": algo_name,
        "loss_cfg": loss_cfg,
        "policy_frame_stack": policy_frame_stack,
    }


def _evaluate_demo_with_context(context: Dict[str, Any], config: Any, demo_id: Optional[str]) -> Dict[str, Any]:
    from robomimic.envs.env_patcher import create_env_patcher

    ckpt_path = context["checkpoint_path"]
    device = context["device"]
    policy = context["policy"]
    ckpt_dict = context["ckpt_dict"]
    cfg = context["cfg"]
    algo_name = context["algo_name"]
    loss_cfg = context["loss_cfg"]
    policy_frame_stack = context["policy_frame_stack"]

    dataset_path = Path(_safe_get(config, "dataset_path")).expanduser()
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    csv_dir = _resolve_output_dir(config, "csv_dir")
    metadata_dir = _resolve_output_dir(config, "metadata_dir", fallback=csv_dir)
    demo_keys = list_dataset_demo_keys(dataset_path)

    requested_stack = _safe_get(config, "frame_stack")
    if requested_stack is not None:
        requested_stack = int(requested_stack)
        if requested_stack <= 0:
            requested_stack = None

    frame_stack = requested_stack or policy_frame_stack or 1
    frame_stack = max(int(frame_stack), 1)

    horizon = _safe_get(config, "horizon")
    horizon = int(horizon) if horizon is not None else None

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

    env = None
    try:
        env = create_env_patcher(
            dataset_path=str(dataset_path),
            demo_id=demo_id,
            frame_stack=frame_stack,
            success_epsilon=float(_safe_get(config, "eps", -1.0)),
            horizon=horizon,
            image_key=image_key,
            pipette_key=pipette_key,
            stage_key=stage_key,
            resistance_key=resistance_key,
            obs_modalities=obs_modalities,
            obs_key_specs=obs_key_specs,
        )
        env_frame_stack = getattr(env, "frame_stack", frame_stack)
        base_env = getattr(env, "base_env", env)
        env_horizon = getattr(base_env, "_horizon", None)
        effective_horizon = int(env_horizon) if env_horizon is not None else int(horizon or 0)

        policy.start_episode()
        env.reset()
        state = env.get_state()
        obs = env.reset_to(state)
        goal = _stack_goal_if_needed(env.get_goal(), env_frame_stack)

        errs: List[float] = []
        rewards: List[float] = []
        acts: List[np.ndarray] = []
        per_step_records: List[Dict[str, Any]] = []
        loss_totals: Dict[str, List[float]] = defaultdict(lambda: [0.0, 0])

        for _ in range(effective_horizon):
            step_idx = env._t
            gt_action = np.asarray(
                env._actions_gt[min(step_idx, env._actions_gt.shape[0] - 1)],
                dtype=np.float32,
            ).reshape(-1)
            start = time.perf_counter()
            act_raw = policy(ob=obs, goal=goal)
            latency_ms = (time.perf_counter() - start) * 1000.0
            act = _extract_policy_action(act_raw)
            if act.shape[0] != gt_action.shape[0]:
                raise ValueError(
                    "[patcherbot_eval] action dimension mismatch: "
                    f"policy {act.shape[0]} vs dataset {gt_action.shape[0]} at step {step_idx}"
                )

            obs, reward, done, info = env.step(act)
            errs.append(info["error_l2"])
            rewards.append(reward)
            acts.append(act)
            per_step_records.append(_build_step_record(step_idx, act, gt_action, info, reward, latency_ms=latency_ms))

            losses = _compute_loss_components(algo_name, loss_cfg, act, gt_action)
            for name, value in losses.items():
                total, count = loss_totals[name]
                loss_totals[name] = [total + float(value), count + 1]

            if done:
                break

        success_info = {}
        try:
            success_info = env.is_success()
        except Exception:
            success_info = {}
        success_flag = bool(success_info.get("task", False))
        steps_taken = len(errs)
        mean_reward = statistics.mean(rewards) if rewards else None
        resolved_demo_key = getattr(env, "_demo_id", demo_id)
        if isinstance(resolved_demo_key, bytes):
            resolved_demo_key = resolved_demo_key.decode("utf-8")
        demo_index = None
        if resolved_demo_key in demo_keys:
            demo_index = demo_keys.index(resolved_demo_key)

        latencies = [record.get("latency_ms") for record in per_step_records if record.get("latency_ms") is not None]
        latency_stats = None
        if latencies:
            latency_stats = {
                "mean": float(statistics.mean(latencies)),
                "median": float(np.median(latencies)),
                "p95": float(np.percentile(latencies, 95)),
            }

        l2_metrics = None
        if errs:
            l2_metrics = {
                "mean": float(statistics.mean(errs)),
                "median": float(np.median(errs)),
                "p95": float(np.percentile(errs, 95)),
            }

        loss_summary = {name: total / count for name, (total, count) in loss_totals.items() if count}
        pos_traj_metrics = None
        if _safe_get(config, "show_pos_traj", True):
            pos_traj_metrics = _compute_positional_trajectory_metrics(cfg, acts, env)

        for record in per_step_records:
            record["checkpoint"] = ckpt_path.name
            record["demo_key"] = resolved_demo_key
            record["demo_id"] = demo_index if demo_index is not None else ""

        csv_path = get_checkpoint_csv_path(str(ckpt_path), csv_dir, demo_key=resolved_demo_key)
        metadata_path = get_checkpoint_metadata_path(str(ckpt_path), metadata_dir, demo_key=resolved_demo_key)
        _write_step_records_csv(csv_path, per_step_records)

        demo_length = getattr(env, "_N", None)
        ckpt_summary = {
            key: ckpt_dict[key]
            for key in ("epoch", "iteration", "global_step", "train_step", "model_epoch")
            if key in ckpt_dict
        }
        train_cfg = getattr(cfg, "train", None)
        config_seed = _safe_get(train_cfg, "seed") if train_cfg is not None else None

        metadata = {
            "timestamp_utc": datetime.utcnow().isoformat(timespec="seconds") + "Z",
            "agent_path": str(ckpt_path),
            "dataset_path": str(dataset_path),
            "demo_id": demo_index,
            "demo_key": resolved_demo_key,
            "num_dataset_demos": len(demo_keys),
            "frame_stack_used": frame_stack,
            "policy_frame_stack": policy_frame_stack,
            "requested_frame_stack": requested_stack,
            "env_frame_stack": env_frame_stack,
            "horizon": effective_horizon,
            "env_horizon": env_horizon,
            "demo_length": demo_length,
            "rollout_steps": steps_taken,
            "success": success_flag,
            "mean_reward": float(mean_reward) if mean_reward is not None else None,
            "algo_name": algo_name,
            "device": str(device),
            "eps": float(_safe_get(config, "eps", -1.0)),
            "per_step_csv": str(csv_path),
            "loss_cfg": _json_safe(loss_cfg) if loss_cfg is not None else None,
            "l2_metrics": l2_metrics,
            "latency_ms_stats": latency_stats,
            "reward_stats": {
                "mean": float(mean_reward) if mean_reward is not None else None,
            },
            "checkpoint_summary": ckpt_summary or None,
            "config_seed": config_seed,
            "loss_summary": loss_summary or None,
            "pos_traj_metrics": pos_traj_metrics,
        }
        metadata = _json_safe(metadata)
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        with metadata_path.open("w", encoding="utf-8") as handle:
            json.dump(metadata, handle, indent=2, sort_keys=True)

        print(f"[RESULT] Checkpoint: {ckpt_path.name}")
        if l2_metrics:
            print(
                f"[RESULT] steps={steps_taken} | mean L2={l2_metrics['mean']:.6f} "
                f"| median={l2_metrics['median']:.6f} | p95={l2_metrics['p95']:.6f}"
            )
        if mean_reward is not None:
            print(f"[RESULT] mean reward={mean_reward:.6f} | success={success_flag}")
        loss_summary_text = _summarize_losses(loss_totals)
        if loss_summary_text:
            print(f"[LOSS] {loss_summary_text}")
        print(f"[RESULT] CSV saved to: {csv_path}")
        print(f"[RESULT] Metadata saved to: {metadata_path}")

        return {
            "success": True,
            "checkpoint": str(ckpt_path),
            "demo_id": demo_index,
            "demo_key": resolved_demo_key,
            "csv_path": str(csv_path),
            "metadata_path": str(metadata_path),
            "per_step_records": per_step_records,
            "metadata": metadata,
            "summary": {
                "steps": steps_taken,
                "success": success_flag,
                "mean_l2": l2_metrics["mean"] if l2_metrics else None,
                "mean_reward": float(mean_reward) if mean_reward is not None else None,
                "l2_metrics": l2_metrics,
                "latency_ms_stats": latency_stats,
                "loss_summary": loss_summary,
                "pos_traj_metrics": pos_traj_metrics,
            },
            "output": f"[RESULT] Checkpoint: {ckpt_path.name}\nsteps={steps_taken}\n",
        }
    finally:
        if env is not None and hasattr(env, "close"):
            try:
                env.close()
            except Exception:
                pass


def evaluate_patcherbot_checkpoint(checkpoint_path: str, config: Any, demo_id: Optional[str] = None) -> Dict[str, Any]:
    context = _load_checkpoint_context(checkpoint_path)
    requested_demo = demo_id if demo_id is not None else _safe_get(config, "demo_id")
    return _evaluate_demo_with_context(context, config, requested_demo)


def evaluate_patcherbot_checkpoint_all_demos(checkpoint_path: str, config: Any) -> List[Dict[str, Any]]:
    context = _load_checkpoint_context(checkpoint_path)
    dataset_path = Path(_safe_get(config, "dataset_path")).expanduser()
    demo_keys = list_dataset_demo_keys(dataset_path)
    return [_evaluate_demo_with_context(context, config, demo_key) for demo_key in demo_keys]


def evaluate_patcherbot_checkpoints(agent_arg: str, config: Any) -> List[Dict[str, Any]]:
    checkpoint_paths = _resolve_checkpoints(agent_arg)
    print(f"[INFO] Found {len(checkpoint_paths)} checkpoints for {agent_arg}")

    results = []
    for ckpt_path in checkpoint_paths:
        print(f"[INFO] Evaluating {ckpt_path.name}...")
        try:
            if _safe_get(config, "evaluate_all_demos", False):
                results.extend(evaluate_patcherbot_checkpoint_all_demos(str(ckpt_path), config))
            else:
                results.append(evaluate_patcherbot_checkpoint(str(ckpt_path), config))
        except Exception as exc:
            print(f"[ERROR] Failed to evaluate {ckpt_path.name}: {exc}")
            results.append(
                {
                    "success": False,
                    "checkpoint": str(ckpt_path),
                    "error": str(exc),
                }
            )
    return results


__all__ = [
    "CSV_PREFIX",
    "METADATA_PREFIX",
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
    "evaluate_patcherbot_checkpoint_all_demos",
    "evaluate_patcherbot_checkpoints",
    "get_checkpoint_csv_path",
    "get_checkpoint_metadata_path",
    "list_dataset_demo_keys",
    "sanitize_demo_name",
    "sanitize_checkpoint_name",
]
