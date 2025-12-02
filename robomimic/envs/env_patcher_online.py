# env_patcher_online.py
from __future__ import annotations

from collections import OrderedDict
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np

import robomimic.envs.env_base as EB


class EnvPatcherOnline(EB.EnvBase):
    """
    Lightweight adapter that mirrors EnvPatcher's observation surface but allows
    external producers (e.g. holypipette) to push live observations and feedback.
    """

    rollout_exceptions: Tuple = ()

    def __init__(
        self,
        *,
        obs_keys: Sequence[str],
        action_dim: Optional[int] = None,
        success_epsilon: float = 0.10,
        modalities: Optional[Mapping[str, str]] = None,
    ) -> None:
        self.frame_stack = 1
        self.success_epsilon = float(success_epsilon)
        self._obs_keys_order = list(obs_keys)
        self._modalities = dict(modalities or {})
        self._action_dim = int(action_dim) if action_dim is not None else None

        self._latest_obs: OrderedDict[str, np.ndarray] = OrderedDict()
        self._latest_reward: float = 0.0
        self._latest_info: Dict[str, Any] = {}
        self._latest_error: Optional[float] = None
        self._latest_error_vec: Optional[np.ndarray] = None
        self._t: int = 0
        self._success_flag: bool = False

    # ------------------------------------------------------------------
    # External data pumps
    # ------------------------------------------------------------------
    def update_observation(
        self,
        *,
        camera_image: Optional[np.ndarray] = None,
        pipette_positions: Optional[np.ndarray] = None,
        stage_positions: Optional[np.ndarray] = None,
        resistance: Optional[np.ndarray] = None,
        extra: Optional[Mapping[str, np.ndarray]] = None,
    ) -> OrderedDict[str, np.ndarray]:
        """
        Accept the raw holypipette observation tuple and store it in an OrderedDict
        keyed to match EnvPatcher conventions.
        """
        obs = OrderedDict()
        if "camera_image" in self._obs_keys_order and camera_image is not None:
            obs["camera_image"] = np.asarray(camera_image, dtype=np.uint8)
        if "pipette_positions" in self._obs_keys_order and pipette_positions is not None:
            obs["pipette_positions"] = np.asarray(pipette_positions, dtype=np.float32)
        if "stage_positions" in self._obs_keys_order and stage_positions is not None:
            obs["stage_positions"] = np.asarray(stage_positions, dtype=np.float32)
        if "resistance" in self._obs_keys_order and resistance is not None:
            obs["resistance"] = np.asarray(resistance, dtype=np.float32)
        if extra:
            for key, value in extra.items():
                if key in self._obs_keys_order:
                    obs[key] = np.asarray(value, dtype=np.float32)
        self._latest_obs = obs
        return self._latest_obs

    def update_feedback(
        self,
        *,
        reward: float = 0.0,
        error_vec: Optional[np.ndarray] = None,
        success: Optional[bool] = None,
        info: Optional[Mapping[str, Any]] = None,
    ) -> None:
        """
        Store feedback to mirror the dataset-backed environment contract.
        """
        self._latest_reward = float(reward)
        self._latest_info = dict(info or {})
        if error_vec is not None:
            err_vec = np.asarray(error_vec, dtype=np.float32).reshape(-1)
            self._latest_error_vec = err_vec
            self._latest_error = float(np.linalg.norm(err_vec, ord=2))
            self._latest_info.setdefault("error_l2", self._latest_error)
            self._latest_info.setdefault("error_vec", err_vec)
        if success is not None:
            self._success_flag = bool(success)
        elif self._latest_error is not None:
            self._success_flag = self._latest_error <= self.success_epsilon

    # ------------------------------------------------------------------
    # EnvBase compatibility
    # ------------------------------------------------------------------
    def reset(self):
        self._t = 0
        if not self._latest_obs:
            raise RuntimeError("Call update_observation before reset() for live mode.")
        return self._latest_obs

    def reset_to(self, state: Mapping[str, Any]):
        self._t = int(state.get("t", 0))
        return self.get_observation()

    def get_observation(self, obs: Optional[Dict[str, np.ndarray]] = None):
        if not self._latest_obs:
            raise RuntimeError("No observation has been pushed into EnvPatcherOnline.")
        return self._latest_obs

    def get_state(self) -> Dict[str, Any]:
        return {"t": int(self._t)}

    def set_state(self, state: Mapping[str, Any]) -> None:
        self._t = int(state.get("t", self._t))

    def step(self, action: np.ndarray):
        """
        Advance the internal clock and echo back the latest observation and feedback.
        """
        act = np.asarray(action, dtype=np.float32).reshape(-1)
        if self._action_dim not in (None, -1) and act.size != self._action_dim:
            raise ValueError(f"Expected action_dim={self._action_dim}, received {act.size}")
        self._t += 1
        obs_next = self.get_observation()
        info = dict(self._latest_info)
        info.setdefault("t", int(self._t))
        return obs_next, float(self._latest_reward), bool(self._success_flag), info

    def is_success(self) -> Dict[str, bool]:
        return {"task": bool(self._success_flag)}
