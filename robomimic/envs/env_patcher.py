# env_patcher.py
from __future__ import annotations
from typing import Dict, Optional, Tuple, Any
import numpy as np
import h5py
import cv2

import robomimic.envs.env_base as EB
from robomimic.envs.wrappers import FrameStackWrapper

class EnvPatcher(EB.EnvBase):
    """
    Dataset-backed environment to mimic the PatcherBot data stream.
    Each step advances one dataset index; reward = -||a_pred - a_gt||_2.

    Observation keys:
        - "camera_image"      : (C,H,W) float32 in [0,1]
        - "pipette_positions" : (3,) float32
        - "stage_positions"   : (3,) float32
        - "resistance"        : () float32
    """

    rollout_exceptions = ()

    def __init__(
        self,
        dataset_path: str,
        *,
        demo_id: Optional[str] = None,
        image_key: str = "camera_image",
        pipette_key: str = "pipette_positions",
        stage_key: str = "stage_positions",
        resistance_key: str = "resistance",
        actions_key: str = "actions",
        crop_hw: Optional[Tuple[int, int]] = None,   # <-- disabled by default
        success_epsilon: float = 0.10,
        horizon: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.dataset_path = dataset_path
        self.image_key = image_key
        self.pipette_key = pipette_key
        self.stage_key = stage_key
        self.resistance_key = resistance_key
        self.actions_key = actions_key
        self.crop_hw = crop_hw
        self.success_epsilon = float(success_epsilon)

        h5 = h5py.File(self.dataset_path, "r")
        if demo_id is None:
            demo_id = sorted(h5["data"].keys())[0]
        self._demo_id = demo_id
        _obs = f"data/{demo_id}/obs"
        _root = f"data/{demo_id}"

        self._images            = h5[f"{_obs}/{self.image_key}"][:]             # (N,H,W,3) uint8
        self._resistance        = h5[f"{_obs}/{self.resistance_key}"][:]        # (N,) or (N,1)
        self._pipette_positions = h5[f"{_obs}/{self.pipette_key}"][:]           # (N,3)
        self._stage_positions   = h5[f"{_obs}/{self.stage_key}"][:]             # (N,3)
        self._actions_gt        = h5[f"{_root}/{self.actions_key}"][:]          # (N, A)

        N = min(len(self._images), len(self._resistance), len(self._pipette_positions),
                len(self._stage_positions), len(self._actions_gt))
        if N <= 0:
            raise RuntimeError("HDF5 contains no samples!")
        self._N = int(N)

        self._t = 0
        self._latest_error = None
        self._latest_error_vec = None
        self._horizon = int(horizon) if horizon is not None else self._N

        self._H, self._W = self._images.shape[1:3]

    @property
    def name(self) -> str:
        return "Patcher"

    @property
    def type(self) -> int:
        # Use a valid builtin type to satisfy the abstract interface for option A.
        # When you register PATCHER_TYPE (Option B), change this to EB.EnvType.PATCHER_TYPE.
        return EB.EnvType.PATCHER_TYPE

    @property
    def base_env(self):
        # No underlying simulator; the dataset-backed env is the base.
        return self

    @property
    def action_dimension(self) -> int:
        return int(self._actions_gt.shape[-1])

    @classmethod
    def create_for_data_processing(
        cls,
        camera_names=None,
        camera_height=None,
        camera_width=None,
        reward_shaping=False,
        render=None,
        render_offscreen=None,
        use_image_obs=None,
        use_depth_obs=None,
        **kwargs,
    ):
        # Expect dataset_path in kwargs; pass everything through.
        return cls(**kwargs)

    def get_goal(self):
        # This dataset-backed env has no separate goal concept.
        return {}

    def set_goal(self, **kwargs):
        # No-op for this env
        return

    def get_reward(self) -> float:
        # Return last step's reward if available, else 0.0 before any step
        if self._latest_error is None:
            return 0.0
        return float(-self._latest_error)

    def is_done(self) -> bool:
        return bool(self._t >= self._horizon)


    def reset(self) -> Dict[str, np.ndarray]:
        self._t = 0
        self._latest_error = None
        self._latest_error_vec = None
        return self.get_observation()

    def get_state(self) -> Dict[str, Any]:
        return {"t": int(self._t), "demo_id": self._demo_id}

    def reset_to(self, state: Dict[str, Any]) -> Dict[str, np.ndarray]:
        self._t = int(state.get("t", 0))
        return self.get_observation()

    def step(self, action: np.ndarray):
        act_pred = np.asarray(action, dtype=np.float32).reshape(-1)
        act_gt   = np.asarray(self._actions_gt[self._t], dtype=np.float32).reshape(-1)

        err_vec = act_pred - act_gt
        err = float(np.linalg.norm(err_vec, ord=2))
        self._latest_error = err
        self._latest_error_vec = err_vec

        self._t += 1
        done = (self._t >= self._horizon) or (self.is_success()["task"])

        obs_next = self.get_observation()
        reward = -err
        info = {"error_l2": err, "error_vec": err_vec.copy(), "t": int(self._t)}
        return obs_next, reward, done, info

    def is_success(self) -> Dict[str, bool]:
        ok = (self._latest_error is not None) and (self._latest_error <= self.success_epsilon)
        return {"task": bool(ok)}

    def get_observation(self, obs: Optional[Dict[str, np.ndarray]] = None) -> Dict[str, np.ndarray]:
        t = min(max(self._t, 0), self._N - 1)
        im_hw3 = self._images[t]  # HWC uint8
        # If you ever need deterministic center cropping for ablations, set crop_hw.
        if self.crop_hw is not None:
            ch, cw = self.crop_hw
            H0, W0 = im_hw3.shape[:2]
            y0 = max(0, (H0 - ch) // 2);  x0 = max(0, (W0 - cw) // 2)
            im_c = im_hw3[y0:y0+ch, x0:x0+cw]
            im_r = cv2.resize(im_c, (W0, H0), interpolation=cv2.INTER_LINEAR)
            im_chw = im_r.astype(np.float32).transpose(2, 0, 1) / 255.0
        else:
            im_chw = im_hw3.astype(np.float32).transpose(2, 0, 1) / 255.0

        pip = self._pipette_positions[t].astype(np.float32).reshape(-1)
        stg = self._stage_positions[t].astype(np.float32).reshape(-1)
        res = np.array(self._resistance[t], dtype=np.float32).reshape(())

        return {
            self.image_key      : im_chw,
            self.pipette_key    : pip,
            self.stage_key      : stg,
            self.resistance_key : res,
        }

    def render(self, mode: str = "human", height: Optional[int] = None,
               width: Optional[int] = None, camera_name: Optional[str] = None, **kwargs):
        if mode == "rgb_array":
            t = min(max(self._t, 0), self._N - 1)
            img = self._images[t]
            if (height is not None) and (width is not None) and (height > 0) and (width > 0):
                img = cv2.resize(img, (width, height), interpolation=cv2.INTER_LINEAR)
            return img
        elif mode == "human":
            img = self.render(mode="rgb_array", height=height or self._H, width=width or self._W)
            cv2.imshow(camera_name or "EnvPatcher", img[:, :, ::-1])
            cv2.waitKey(1)
            return None
        else:
            return None

    def serialize(self) -> Dict[str, Any]:
        return dict(
            env_name="Patcher",
            type="PATCHER_TYPE",
            env_kwargs=dict(
                dataset_path=self.dataset_path,
                demo_id=self._demo_id,
                image_key=self.image_key,
                pipette_key=self.pipette_key,
                stage_key=self.stage_key,
                resistance_key=self.resistance_key,
                actions_key=self.actions_key,
                crop_hw=self.crop_hw,
                success_epsilon=self.success_epsilon,
                horizon=self._horizon,
            ),
        )

def create_env_patcher(dataset_path: str, *, frame_stack: int = 1, **kwargs):
    base = EnvPatcher(dataset_path=dataset_path, **kwargs)
    if frame_stack and frame_stack > 1:
        return FrameStackWrapper(base, num_frames=frame_stack)
    return base
