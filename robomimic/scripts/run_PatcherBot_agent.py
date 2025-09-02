# scripts/run_PatcherBot_agent.py
import argparse, numpy as np, statistics
from pathlib import Path

import robomimic.utils.file_utils as FileUtils
import robomimic.utils.torch_utils as TorchUtils

from robomimic.envs.env_patcher import create_env_patcher

agent = r"C:\Users\sa-forest\Documents\GitHub\robomimic\bc_patcherBot\v0_029\20250831172330\models\model_epoch_50.pth"
dataset = r"C:\\Users\\sa-forest\\Documents\\GitHub\\holypipette-pbl\\holypipette\\deepLearning\\patchModel\\test_data\\HEKHUNTER_inference_set3.hdf5"
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--agent", required=False, default = agent,  help="Path to .pth checkpoint")
    ap.add_argument("--dataset", required=False,default = dataset,  help="Path to .hdf5")
    ap.add_argument("--horizon", type=int, default=400)         # matches rollout.horizon
    ap.add_argument("--frame_stack", type=int, default=1)       # RNN policy => no frame stacking
    ap.add_argument("--eps", type=float, default=0.10)
    args = ap.parse_args()

    device = TorchUtils.get_torch_device(try_to_use_cuda=True)
    policy, _ = FileUtils.policy_from_checkpoint(ckpt_path=args.agent, device=device, verbose=True)

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

    errs, rewards = [], []
    for _ in range(args.horizon):
        act = policy(ob=obs)
        obs, r, done, info = env.step(act)
        errs.append(info["error_l2"]); rewards.append(r)
        if done:
            break

    print(f"[RESULT] steps={len(errs)} | mean L2={statistics.mean(errs):.6f} | "
          f"median={np.median(errs):.6f} | p95={np.percentile(errs,95):.6f}")
    print(f"[RESULT] mean reward={statistics.mean(rewards):.6f} | success={env.is_success()['task']}")

if __name__ == "__main__":
    main()
