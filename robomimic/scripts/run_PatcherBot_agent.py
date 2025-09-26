# scripts/run_PatcherBot_agent.py
import argparse
import csv
import json
import statistics
from collections import defaultdict
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence
from pathlib import Path
from datetime import datetime

import h5py
import numpy as np

import robomimic.utils.file_utils as FileUtils
import robomimic.utils.torch_utils as TorchUtils

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





def _append_vector(prefix: str, array: Optional[Iterable], record: Dict[str, Any]) -> None:
    if array is None:
        return
    arr = np.asarray(array, dtype=np.float32).reshape(-1)
    for idx, value in enumerate(arr):
        record[f'{prefix}_{idx}'] = float(value)


def _build_step_record(step_idx: int, act_pred: np.ndarray, act_gt: np.ndarray, info: Mapping[str, Any], reward: float) -> Dict[str, Any]:
    record: Dict[str, Any] = {
        'step': int(step_idx),
        'error_l2': float(info.get('error_l2', np.nan)),
    }
    t_val = info.get('t')
    if t_val is not None:
        record['env_t'] = int(t_val)
    record['reward'] = float(reward)
    _append_vector('gt', act_gt, record)
    _append_vector('pred', act_pred, record)
    err_vec = info.get('error_vec')
    _append_vector('err', err_vec, record)
    if err_vec is not None:
        _append_vector('abs_err', np.abs(err_vec), record)
    return record


def _json_safe(value: Any):
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

def _compute_loss_components(algo_name: str, loss_cfg: Optional[Mapping[str, Any]], act_pred: np.ndarray, act_gt: np.ndarray) -> Dict[str, float]:
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


def _summarize_losses(loss_totals: Dict[str, list]) -> str:
    parts = []
    for name, (total, count) in sorted(loss_totals.items()):
        if count:
            parts.append(f"{name}={total / count:.6f}")
    return " | ".join(parts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--agent", required=False, default = r"C:\Users\sa-forest\Documents\GitHub\robomimic\df_patcherBot\PipetteFinding\v0_003\20250926082848\last.pth",  help="Path to .pth checkpoint")
    ap.add_argument("--dataset", required=False,default = r"C:\Users\sa-forest\Documents\GitHub\holypipette-pbl\experiments\Datasets\PatcherBot_dataset_v0_002\PatcherBot_dataset_v0_002_find_pipette.hdf5",  help="Path to .hdf5")
    ap.add_argument("--horizon", type=int, default=None)
    ap.add_argument("--frame_stack", type=int, default=None, help="Frame stack override; defaults to policy config")
    ap.add_argument("--eps", type=float, default=-1.0)
    ap.add_argument("--show_pos_traj", action="store_true", default = True,help="compute positional trajectory error like HuntTester")
    ap.add_argument("--per_step_csv", type=str, default= r"C:\Users\sa-forest\Documents\GitHub\robomimic\df_patcherBot\PipetteFinding\results\results_df_PatcherBot_v0_003.csv", help="Optional path to save per-step action errors as CSV")
    ap.add_argument("--rollout_metadata",type= str, default= r"C:\Users\sa-forest\Documents\GitHub\robomimic\df_patcherBot\PipetteFinding\results\metadata_df_PatcherBot_v0_003.json", help="Optional path to save rollout metadata as JSON")
    args = ap.parse_args()

    device = TorchUtils.get_torch_device(try_to_use_cuda=True)
    policy, ckpt_dict = FileUtils.policy_from_checkpoint(ckpt_path=args.agent, device=device, verbose=True)
    cfg, _ = FileUtils.config_from_checkpoint(ckpt_path=args.agent, ckpt_dict=ckpt_dict)

    algo_name = str(ckpt_dict.get("algo_name", "")).lower()
    print(f"[INFO] loaded algorithm: {algo_name or 'unknown'}")

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

    requested_stack = args.frame_stack if (args.frame_stack is not None and args.frame_stack > 0) else None
    frame_stack = requested_stack or policy_frame_stack or 1
    frame_stack = max(int(frame_stack), 1)
    if requested_stack and policy_frame_stack and requested_stack != policy_frame_stack:
        print(f"[WARN] overriding policy frame_stack={policy_frame_stack} with requested value {requested_stack}")
    else:
        print(f"[INFO] using frame_stack={frame_stack} (policy default: {policy_frame_stack or 'n/a'})")

    if args.horizon is None:
        with h5py.File(args.dataset, "r") as h5:
            demo_id = sorted(h5["data"].keys())[2] # make sure it is same in env_patcher.py for now
            args.horizon = h5["data"][demo_id]["actions"].shape[0]
        print(f"[INFO] setting horizon to {args.horizon} from demo length")

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

    env = create_env_patcher(
        dataset_path=args.dataset,
        frame_stack=frame_stack,
        success_epsilon=args.eps,
        horizon=args.horizon,
        image_key=image_key,
        pipette_key=pipette_key,
        stage_key=stage_key,
        resistance_key=resistance_key,
        obs_modalities=obs_modalities,
        obs_key_specs=obs_key_specs,
    )
    env_frame_stack = getattr(env, "frame_stack", frame_stack)

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

    policy.start_episode()
    obs = env.reset()
    state = env.get_state()
    obs = env.reset_to(state)
    goal = env.get_goal()
    goal = _stack_goal_if_needed(goal, env_frame_stack)

    errs, rewards, acts = [], [], []
    per_step_records = []
    loss_totals: Dict[str, list] = defaultdict(lambda: [0.0, 0])
    for _ in range(args.horizon):
        step_idx = env._t
        gt_action = np.asarray(env._actions_gt[min(step_idx, env._actions_gt.shape[0] - 1)], dtype=np.float32).reshape(-1)
        act_raw = policy(ob=obs, goal=goal)
        act = _extract_policy_action(act_raw)
        print(f"act={act}")
        obs, r, done, info = env.step(act)
        errs.append(info["error_l2"])
        rewards.append(r)
        acts.append(act)
        per_step_records.append(_build_step_record(step_idx, act, gt_action, info, r))
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
    l2_metrics = None
    if errs:
        l2_metrics = {
            "mean": float(statistics.mean(errs)),
            "median": float(np.median(errs)),
            "p95": float(np.percentile(errs, 95)),
        }
        print(f"[RESULT] steps={steps_taken} | mean L2={l2_metrics['mean']:.6f} | median={l2_metrics['median']:.6f} | p95={l2_metrics['p95']:.6f}")
        if mean_reward is not None:
            print(f"[RESULT] mean reward={mean_reward:.6f} | success={success_flag}")
    elif mean_reward is not None:
        print(f"[RESULT] mean reward={mean_reward:.6f} | success={success_flag}")
    if loss_totals:
        print(f"[LOSS] {_summarize_losses(loss_totals)}")

    per_step_csv_path = None
    if args.per_step_csv and per_step_records:
        csv_path = Path(args.per_step_csv).expanduser()
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = []
        for record in per_step_records:
            for key in record:
                if key not in fieldnames:
                    fieldnames.append(key)
        with csv_path.open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for record in per_step_records:
                writer.writerow({key: record.get(key, '') for key in fieldnames})
        per_step_csv_path = csv_path
        print(f"[RESULT] wrote per-step errors to {csv_path}")

    metadata_path = getattr(args, 'rollout_metadata', None)
    if metadata_path:
        meta_path = Path(metadata_path).expanduser()
        meta_path.parent.mkdir(parents=True, exist_ok=True)
        env_horizon = getattr(env, '_horizon', None)
        demo_length = getattr(env, '_N', None)
        ckpt_summary = {k: ckpt_dict[k] for k in ('epoch', 'iteration', 'global_step', 'train_step', 'model_epoch') if k in ckpt_dict}
        train_cfg = getattr(cfg, 'train', None)
        config_seed = _safe_get(train_cfg, 'seed') if train_cfg is not None else None
        demo_key = getattr(env, '_demo_id', None)
        if isinstance(demo_key, bytes):
            demo_key = demo_key.decode('utf-8')
        demo_index = None
        if demo_key is not None:
            try:
                with h5py.File(args.dataset, 'r') as h5_demo:
                    demo_keys = list(h5_demo['data'].keys())
                decoded_keys = [k.decode('utf-8') if isinstance(k, bytes) else k for k in demo_keys]
                sorted_keys = sorted(decoded_keys)
                if demo_key in sorted_keys:
                    demo_index = sorted_keys.index(demo_key)
            except Exception:
                demo_index = None
        metadata = {
            'timestamp_utc': datetime.utcnow().isoformat(timespec='seconds') + 'Z',
            'agent_path': str(Path(args.agent).expanduser()),
            'dataset_path': str(Path(args.dataset).expanduser()),
            'demo_id': demo_index,
            'demo_key': demo_key,
            'frame_stack_used': frame_stack,
            'policy_frame_stack': policy_frame_stack,
            'requested_frame_stack': requested_stack,
            'env_frame_stack': getattr(env, 'frame_stack', frame_stack),
            'horizon': int(args.horizon) if args.horizon is not None else None,
            'env_horizon': env_horizon,
            'demo_length': demo_length,
            'rollout_steps': steps_taken,
            'success': success_flag,
            'mean_reward': float(mean_reward) if mean_reward is not None else None,
            'algo_name': algo_name,
            'device': str(device),
            'eps': float(args.eps) if args.eps is not None else None,
            'per_step_csv': str(per_step_csv_path) if per_step_csv_path else None,
            'loss_cfg': _json_safe(loss_cfg) if loss_cfg is not None else None,
            'l2_metrics': l2_metrics,
            'reward_stats': {
                'mean': float(mean_reward) if mean_reward is not None else None,
            },
            'checkpoint_summary': ckpt_summary or None,
            'config_seed': config_seed,
        }
        metadata = _json_safe(metadata)
        with meta_path.open('w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, sort_keys=True)
        print(f"[RESULT] wrote rollout metadata to {meta_path}")

    if args.show_pos_traj:
        try:
            seq_len = int(getattr(cfg.train, 'seq_length', 16))
        except Exception:
            seq_len = 16
        A = np.asarray(acts).reshape(len(acts), -1)
        if A.shape[1] >= 6:
            pred_deltas = A[:, 3:6]
            start = max(0, seq_len - 1)
            end = min(start + pred_deltas.shape[0], len(env._pipette_positions))
            init_pos = env._pipette_positions[start]
            obs_pos = env._pipette_positions[start:end]
            n = min(pred_deltas.shape[0], obs_pos.shape[0])
            pred_deltas = pred_deltas[:n]
            obs_pos = obs_pos[:n]
            pred_pos = [init_pos]
            for d in pred_deltas:
                pred_pos.append(pred_pos[-1] + d)
            pred_pos = np.stack(pred_pos[1:])
            err = np.linalg.norm(pred_pos - obs_pos, axis=1)
            print(f"[POS_ERR] steps={len(err)} | mean={err.mean():.6f} | median={np.median(err):.6f} | p95={np.percentile(err,95):.6f}")
        else:
            print("[POS_ERR] Skipped: action dimension < 6")


if __name__ == "__main__":
    main()
