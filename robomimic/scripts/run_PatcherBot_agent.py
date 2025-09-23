# scripts/run_PatcherBot_agent.py
import argparse
import numpy as np 
import statistics
from pathlib import Path
import h5py

import robomimic.utils.file_utils as FileUtils
import robomimic.utils.torch_utils as TorchUtils

from robomimic.envs.env_patcher import create_env_patcher

# agent = r"C:\Users\sa-forest\Documents\GitHub\robomimic\bc_patcherBot\v0_029\20250831172330\models\model_epoch_50.pth"
# agent = r"C:\Users\sa-forest\Documents\GitHub\robomimic\bc_patcherBot\v0_031\20250915183951\models\model_epoch_50.pth"
# agent = r"C:\Users\sa-forest\Documents\GitHub\robomimic\bc_patcherBot\v0_034\20250916172830\last.pth"
# agent = r"C:\Users\sa-forest\Documents\GitHub\robomimic\bc_patcherBot\v0_035\20250917152411\last.pth"
# agent = r"C:\Users\sa-forest\Documents\GitHub\robomimic\bc_patcherBot\v0_030\20250915140359\last.pth"
# agent = r"C:\Users\sa-forest\Documents\GitHub\robomimic\bc_patcherBot\v0_036\20250918011750\last.pth"
# agent = r"C:\Users\sa-forest\Documents\GitHub\robomimic\bc_patcherBot\v0_037\20250921164912\last.pth"
# version 38 testing
# agent = r"C:\Users\sa-forest\Documents\GitHub\robomimic\bc_patcherBot\v0_038\20250921221249\models\model_epoch_500.pth"
# agent = r"C:\Users\sa-forest\Documents\GitHub\robomimic\df_patcherBot\v0_002\20250923065914\models\model_epoch_300.pth"
# agent = r"C:\Users\sa-forest\Documents\GitHub\robomimic\bc_patcherBot\v0_040\20250923123226\models\model_epoch_25.pth"
agent = r"C:\Users\sa-forest\Documents\GitHub\robomimic\df_patcherBot\v0_003\20250923163048\last.pth"
# dataset = r"C:\\Users\\sa-forest\\Documents\\GitHub\\holypipette-pbl\\holypipette\\deepLearning\\patchModel\\test_data\\HEKHUNTER_inference_set3.hdf5"
dataset = r"c:\Users\sa-forest\Documents\GitHub\holypipette-pbl\holypipette\deepLearning\patchModel\test_data\HEKHUNTER_inference_set_goal5.hdf5"
# dataset = r"C:\Users\sa-forest\Documents\GitHub\holypipette-pbl\holypipette\deepLearning\patchModel\test_data\HEKHUNTER_sanity_set_goal3.hdf5"
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--agent", required=False, default = agent,  help="Path to .pth checkpoint")
    ap.add_argument("--dataset", required=False,default = dataset,  help="Path to .hdf5")
    ap.add_argument("--horizon", type=int, default=None)         # matches rollout.horizon
    ap.add_argument("--frame_stack", type=int, default=None, help="Frame stack override; defaults to policy config")
    ap.add_argument("--eps", type=float, default=-1.0)
    ap.add_argument("--show_pos_traj", action="store_true", default = True,help="compute positional trajectory error like HuntTester")
    args = ap.parse_args()

    device = TorchUtils.get_torch_device(try_to_use_cuda=True)
    policy, ckpt_dict = FileUtils.policy_from_checkpoint(ckpt_path=args.agent, device=device, verbose=True)

    algo_name = str(ckpt_dict.get("algo_name", "")).lower()
    print(f"[INFO] loaded algorithm: {algo_name or 'unknown'}")

    policy_frame_stack = None
    policy_impl = getattr(policy, "policy", None)
    if policy_impl is not None:
        cfg = getattr(policy_impl, "global_config", None)
        if cfg is not None:
            try:
                cfg_stack = getattr(cfg.train, "frame_stack", None)
                if cfg_stack is not None:
                    policy_frame_stack = int(cfg_stack)
            except AttributeError:
                policy_frame_stack = None

    if policy_frame_stack is not None and policy_frame_stack <= 0:
        policy_frame_stack = None

    requested_stack = args.frame_stack if (args.frame_stack is not None and args.frame_stack > 0) else None
    frame_stack = requested_stack or policy_frame_stack or 1
    frame_stack = max(int(frame_stack), 1)
    if requested_stack and policy_frame_stack and requested_stack != policy_frame_stack:
        print(f"[WARN] overriding policy frame_stack={policy_frame_stack} with requested value {requested_stack}")
    else:
        print(f"[INFO] using frame_stack={frame_stack} (policy default: {policy_frame_stack or 'n/a'})")

    # define horizon from demonstration length if not specified
    if args.horizon is None:
        with h5py.File(args.dataset, "r") as h5:
            demo_id = sorted(h5["data"].keys())[0]
            args.horizon = h5["data"][demo_id]["actions"].shape[0]
        print(f"[INFO] setting horizon to {args.horizon} from demo length")

    env = create_env_patcher(
        dataset_path=args.dataset,
        frame_stack=frame_stack,
        success_epsilon=args.eps,
        horizon=args.horizon,
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
    for _ in range(args.horizon):
        act = policy(ob=obs, goal=goal)
        print(f"act={act}")
        obs, r, done, info = env.step(act)
        errs.append(info["error_l2"]); rewards.append(r); acts.append(act)
        if done:
            break

    print(f"[RESULT] steps={len(errs)} | mean L2={statistics.mean(errs):.6f} | "
          f"median={np.median(errs):.6f} | p95={np.percentile(errs,95):.6f}")
    print(f"[RESULT] mean reward={statistics.mean(rewards):.6f} | success={env.is_success()['task']}")

    if args.show_pos_traj:
        # integrate predicted Δxyz to absolute positions and compare to observed
        try:
            from robomimic.utils.file_utils import config_from_checkpoint
            cfg, _ = config_from_checkpoint(ckpt_path=args.agent)
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
            pred_deltas = pred_deltas[:n]; obs_pos = obs_pos[:n]
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

