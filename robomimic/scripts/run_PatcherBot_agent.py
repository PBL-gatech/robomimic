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
agent = r"C:\Users\sa-forest\Documents\GitHub\robomimic\bc_patcherBot\v0_034\20250916172830\last.pth"
# agent = r"C:\Users\sa-forest\Documents\GitHub\robomimic\bc_patcherBot\v0_030\20250915140359\last.pth"
# dataset = r"C:\\Users\\sa-forest\\Documents\\GitHub\\holypipette-pbl\\holypipette\\deepLearning\\patchModel\\test_data\\HEKHUNTER_inference_set3.hdf5"
dataset = r"c:\Users\sa-forest\Documents\GitHub\holypipette-pbl\holypipette\deepLearning\patchModel\test_data\HEKHUNTER_inference_set_goal2.hdf5"
# dataset = r"C:\Users\sa-forest\Documents\GitHub\holypipette-pbl\holypipette\deepLearning\patchModel\test_data\HEKHUNTER_sanity_set_goal.hdf5"
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--agent", required=False, default = agent,  help="Path to .pth checkpoint")
    ap.add_argument("--dataset", required=False,default = dataset,  help="Path to .hdf5")
    ap.add_argument("--horizon", type=int, default=None)         # matches rollout.horizon
    ap.add_argument("--frame_stack", type=int, default=1)       # RNN policy => no frame stacking
    ap.add_argument("--eps", type=float, default=-1.0)
    ap.add_argument("--show_pos_traj", action="store_true", default = True,help="compute positional trajectory error like HuntTester")
    args = ap.parse_args()

    device = TorchUtils.get_torch_device(try_to_use_cuda=True)
    policy, _ = FileUtils.policy_from_checkpoint(ckpt_path=args.agent, device=device, verbose=True)

    # define horizon from demonstration length if not specified
    if args.horizon is None:
        with h5py.File(args.dataset, "r") as h5:
            demo_id = sorted(h5["data"].keys())[0]
            args.horizon = h5["data"][demo_id]["actions"].shape[0]
        print(f"[INFO] setting horizon to {args.horizon} from demo length")

    env = create_env_patcher(
        dataset_path=args.dataset,
        frame_stack=args.frame_stack,
        success_epsilon=args.eps,
        horizon=args.horizon,
    )

    policy.start_episode()
    obs = env.reset()
    state = env.get_state()
    obs = env.reset_to(state)
    goal = env.get_goal()

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
