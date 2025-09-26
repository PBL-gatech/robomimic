# env_patcher.py
from __future__ import annotations

from collections import OrderedDict
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple
import warnings

import cv2
import h5py
import numpy as np

import robomimic.envs.env_base as EB
from robomimic.envs.wrappers import FrameStackWrapper


class EnvPatcher(EB.EnvBase):
    """
    Dataset-backed environment to mimic the PatcherBot data stream.
    Each step advances one dataset index; reward = -||a_pred - a_gt||_2.

    Observation keys:
        - "camera_image"      : (H,W,3) uint8 (ObsUtils will CHW + normalize)
        - "pipette_positions" : (3,) float32
        - "stage_positions"   : (3,) float32
        - "resistance"        : () float32
    """

    rollout_exceptions = ()

    def __init__(
        self,
        dataset_path: str = None,
        *,
        # standard EnvBase ctor params (accepted for compatibility; unused)
        env_name: Optional[str] = None,
        render: bool = False,
        render_offscreen: bool = False,
        use_image_obs: bool = True,
        use_depth_obs: bool = False,
        lang: Optional[str] = None,
        # patcher-specific
        demo_id: Optional[str] = None,
        image_key: Optional[str] = "camera_image",
        pipette_key: Optional[str] = "pipette_positions",
        stage_key: Optional[str] = "stage_positions",
        resistance_key: Optional[str] = "resistance",
        actions_key: str = "actions",
        crop_hw: Optional[Tuple[int, int]] = None,
        success_epsilon: float = 0.10,
        frame_stack: int = 1,
        horizon: Optional[int] = None,
        obs_modalities: Optional[Mapping[str, Sequence[str]]] = None,
        obs_key_specs: Optional[Mapping[str, Dict[str, Any]]] = None,
        auto_generate_missing: bool = True,
        **kwargs,
    ) -> None:
        # Note: follow EnvGym / EnvRobosuite pattern - do not call EnvBase.__init__
        self.dataset_path = dataset_path
        self.image_key = image_key
        self.pipette_key = pipette_key
        self.stage_key = stage_key
        self.resistance_key = resistance_key
        self.actions_key = actions_key
        self.crop_hw = crop_hw
        self.success_epsilon = float(success_epsilon)
        self.frame_stack = int(frame_stack) if frame_stack else 1
        self.auto_generate_missing = bool(auto_generate_missing)

        self.obs_key_specs: Dict[str, Dict[str, Any]] = {
            str(k): dict(v) for k, v in (obs_key_specs.items() if obs_key_specs else [])
        }
        self._obs_modalities_map: Dict[str, str] = self._build_modality_map(obs_modalities)
        self._obs_keys_order = self._collect_requested_obs_keys()

        with h5py.File(self.dataset_path, "r") as h5:
            if demo_id is None:
                demo_id = sorted(h5["data"].keys())[0]
            self._demo_id = demo_id
            print(f"[EnvPatcher] loading demo_id={self._demo_id} from {self.dataset_path}")
            obs_group_path = f"data/{demo_id}/obs"
            root_group_path = f"data/{demo_id}"
            obs_group = h5[obs_group_path]
            self._actions_gt = np.asarray(h5[f"{root_group_path}/{self.actions_key}"][:], dtype=np.float32)
            if self._actions_gt.ndim < 2:
                raise ValueError("actions dataset must be 2D (timesteps x action_dim)")
            self._N = int(self._actions_gt.shape[0])

            obs_arrays: Dict[str, np.ndarray] = {}
            resolved_specs: Dict[str, Dict[str, Any]] = {}
            dataset_keys = set(obs_group.keys())
            missing_keys = []

            for key in self._obs_keys_order:
                if key in dataset_keys:
                    data = np.asarray(obs_group[key][:])
                    spec, data = self._spec_from_array(key, data)
                    data = self._trim_or_pad(data, spec["fill_value"])
                    obs_arrays[key] = data
                    resolved_specs[key] = spec
                else:
                    missing_keys.append(key)

            generated_keys = set()
            for key in missing_keys:
                spec = self._resolve_spec_from_hints(key, resolved_specs)
                if spec is None:
                    if not self.auto_generate_missing:
                        raise KeyError(f"Missing observation key '{key}' and auto generation disabled")
                    warnings.warn(
                        f"[{key}] missing in dataset; using scalar zero placeholder",
                        RuntimeWarning,
                    )
                    spec = dict(shape=(1,), dtype=np.float32, fill_value=0.0, modality=self._obs_modalities_map.get(key))
                arr = self._generate_array_for_key(key, spec, obs_arrays)
                obs_arrays[key] = arr
                resolved_specs[key] = spec
                generated_keys.add(key)

        self._obs_arrays = obs_arrays
        self._resolved_obs_specs = resolved_specs
        self._generated_obs_keys = generated_keys
        self._dataset_obs_keys = dataset_keys

        self._actions_gt = self._trim_or_pad(self._actions_gt, 0.0)
        self._t = 0
        self._latest_error: Optional[float] = None
        self._latest_error_vec: Optional[np.ndarray] = None
        self._horizon = int(horizon) if horizon is not None else self._N
        self._goal_index = min(max(self._horizon - 1, 0), self._N - 1)

        self._images = self._ensure_array_presence(self.image_key, default_channels=3, default_dtype=np.uint8)
        self._pipette_positions = self._ensure_array_presence(self.pipette_key, default_channels=3)
        self._stage_positions = self._ensure_array_presence(self.stage_key, default_channels=3)
        self._resistance = self._ensure_array_presence(self.resistance_key, default_channels=1)

        if self._images is not None and self._images.ndim >= 4:
            self._H, self._W = self._images.shape[1:3]
        else:
            self._H = self._W = 0

    def _build_modality_map(self, modalities: Optional[Mapping[str, Sequence[str]]]) -> Dict[str, str]:
        mapping: Dict[str, str] = {}
        if modalities is not None:
            for modality, keys in modalities.items():
                if keys is None:
                    continue
                if isinstance(keys, (str, bytes)):
                    iter_keys: Iterable[str] = [keys]
                else:
                    iter_keys = list(keys)
                for key in iter_keys:
                    mapping[str(key)] = str(modality)
        if self.image_key and self.image_key not in mapping:
            mapping[self.image_key] = "rgb"
        for base_key in (self.pipette_key, self.stage_key, self.resistance_key):
            if base_key and base_key not in mapping:
                mapping[base_key] = "low_dim"
        return mapping

    def _collect_requested_obs_keys(self) -> Sequence[str]:
        ordered: list[str] = []
        seen = set()
        for key in list(self._obs_modalities_map.keys()) + [
            self.image_key,
            self.pipette_key,
            self.stage_key,
            self.resistance_key,
        ] + list(self.obs_key_specs.keys()):
            if key and key not in seen:
                ordered.append(key)
                seen.add(key)
        return ordered

    def _spec_from_array(self, key: str, array: np.ndarray) -> Tuple[Dict[str, Any], np.ndarray]:
        modality = self._obs_modalities_map.get(key)
        data = np.asarray(array)
        if data.shape[0] == 0:
            raise ValueError(f"Observation '{key}' is empty in dataset")
        if modality in (None, "low_dim", "scan") and data.dtype != np.float32:
            data = data.astype(np.float32)
        spec = dict(
            shape=tuple(int(s) for s in data.shape[1:]),
            dtype=data.dtype,
            fill_value=0 if data.dtype == np.uint8 else 0.0,
            modality=modality,
            source="dataset",
        )
        return spec, data

    def _resolve_spec_from_hints(
        self,
        key: str,
        resolved_specs: Mapping[str, Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        hints = dict(self.obs_key_specs.get(key, {}))
        modality = hints.get("modality") or self._obs_modalities_map.get(key)
        hints["modality"] = modality
        dtype = hints.get("dtype")
        if dtype is None:
            dtype = np.uint8 if modality == "rgb" else np.float32
        hints["dtype"] = np.dtype(dtype)
        if "fill_value" not in hints:
            hints["fill_value"] = 0 if hints["dtype"] == np.uint8 else 0.0
        if "shape" not in hints:
            ref_key = hints.get("like")
            if ref_key and ref_key in resolved_specs:
                hints["shape"] = resolved_specs[ref_key]["shape"]
            else:
                guessed = self._guess_shape_from_modality(modality, hints)
                if guessed is None:
                    return None
                hints["shape"] = guessed
        hints["shape"] = tuple(int(s) for s in hints["shape"])
        return hints

    def _guess_shape_from_modality(self, modality: Optional[str], hints: Dict[str, Any]) -> Optional[Tuple[int, ...]]:
        if modality == "rgb":
            if "shape" in hints:
                return tuple(int(s) for s in hints["shape"])
            hw = hints.get("hw") or hints.get("crop") or self.crop_hw
            if hw is None:
                height = int(hints.get("height", 84))
                width = int(hints.get("width", 84))
            else:
                height, width = int(hw[0]), int(hw[1])
            channels = int(hints.get("channels", 3))
            return (height, width, channels)
        if modality == "depth":
            hw = hints.get("hw") or self.crop_hw
            if hw is None:
                height = int(hints.get("height", 84))
                width = int(hints.get("width", 84))
            else:
                height, width = int(hw[0]), int(hw[1])
            channels = int(hints.get("channels", 1))
            return (height, width, channels)
        dim = hints.get("dim")
        if isinstance(dim, int):
            return (int(dim),)
        if isinstance(dim, Sequence):
            return tuple(int(s) for s in dim)
        return (1,)

    def _generate_array_for_key(
        self,
        key: str,
        spec: Dict[str, Any],
        obs_arrays: Mapping[str, np.ndarray],
    ) -> np.ndarray:
        generator = spec.get("generator")
        arr: Optional[np.ndarray] = None
        if callable(generator):
            arr = generator(self, spec, obs_arrays)
        elif spec.get("transform") == "diff":
            source_key = spec.get("source_key")
            if source_key and source_key in obs_arrays:
                base = np.asarray(obs_arrays[source_key], dtype=np.float32)
                diff = np.diff(base, axis=0, prepend=base[:1])
                arr = diff.astype(spec["dtype"])
        if arr is None:
            arr = np.full((self._N,) + tuple(spec["shape"]), spec["fill_value"], dtype=spec["dtype"])
        else:
            arr = np.asarray(arr, dtype=spec["dtype"])
            arr = self._trim_or_pad(arr, spec["fill_value"])
            desired = (self._N,) + tuple(spec["shape"])
            if arr.shape != desired:
                try:
                    arr = np.broadcast_to(arr, desired).astype(spec["dtype"], copy=False)
                except ValueError:
                    warnings.warn(
                        f"[{key}] generated shape {arr.shape} cannot broadcast to {desired}; using fill value",
                        RuntimeWarning,
                    )
                    arr = np.full(desired, spec["fill_value"], dtype=spec["dtype"])
        return arr

    def _ensure_array_presence(
        self,
        key: Optional[str],
        *,
        default_channels: int,
        default_dtype: np.dtype = np.float32,
    ) -> Optional[np.ndarray]:
        if key is None:
            return None
        arr = self._obs_arrays.get(key)
        if arr is not None:
            return arr
        spec = self._resolved_obs_specs.get(key)
        if spec is None:
            dtype = default_dtype
            shape = (default_channels,)
            fill = 0 if dtype == np.uint8 else 0.0
        else:
            dtype = spec["dtype"]
            shape = spec["shape"]
            fill = spec.get("fill_value", 0 if dtype == np.uint8 else 0.0)
        arr = np.full((self._N,) + tuple(shape), fill, dtype=dtype)
        self._obs_arrays[key] = arr
        self._resolved_obs_specs[key] = dict(shape=tuple(shape), dtype=dtype, fill_value=fill, modality=self._obs_modalities_map.get(key))
        self._generated_obs_keys.add(key)
        return arr

    def _trim_or_pad(self, array: np.ndarray, fill_value: float) -> np.ndarray:
        arr = np.asarray(array)
        if arr.shape[0] < self._N:
            pad_shape = (self._N - arr.shape[0],) + arr.shape[1:]
            pad = np.full(pad_shape, fill_value, dtype=arr.dtype)
            arr = np.concatenate([arr, pad], axis=0)
        elif arr.shape[0] > self._N:
            arr = arr[:self._N]
        return arr

    @property
    def name(self) -> str:
        return "Patcher"

    @property
    def type(self) -> int:
        return EB.EnvType.PATCHER_TYPE

    @property
    def base_env(self):
        return self

    @property
    def action_dimension(self) -> int:
        return int(self._actions_gt.shape[-1])

    @classmethod
    def create_for_data_processing(
        cls,
        env_name=None,
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
        return cls(env_name=env_name, render=bool(render), render_offscreen=bool(render_offscreen), use_image_obs=bool(use_image_obs), use_depth_obs=bool(use_depth_obs), **kwargs)

    def set_seed(self, seed: Optional[int] = None):
        return

    def __repr__(self):
        return f"EnvPatcher(demo_id={self._demo_id}, N={self._N})"

    def get_goal(self):
        return self._observation_at_index(self._goal_index)

    def set_goal(self, **kwargs):
        idx = None
        if "index" in kwargs:
            idx = kwargs["index"]
        elif "t" in kwargs:
            idx = kwargs["t"]
        if idx is None:
            return
        idx = int(idx)
        self._goal_index = min(max(idx, 0), self._N - 1)
        return

    def get_reward(self) -> float:
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
        t = min(max(self._t, 0), self._N - 1)
        pip = self._pipette_positions[t].astype(np.float32, copy=False).reshape(-1)
        stg = self._stage_positions[t].astype(np.float32, copy=False).reshape(-1)
        res = np.asarray(self._resistance[t], dtype=np.float32).reshape(-1)
        states = np.concatenate([pip, stg, res], axis=0)
        return {"t": int(self._t), "demo_id": self._demo_id, "states": states}

    def reset_to(self, state: Dict[str, Any]) -> Dict[str, np.ndarray]:
        self._t = int(state.get("t", 0))
        return self.get_observation()

    def step(self, action: np.ndarray):
        act_pred = np.asarray(action, dtype=np.float32).reshape(-1)
        act_gt = np.asarray(self._actions_gt[self._t], dtype=np.float32).reshape(-1)

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
        return self._observation_at_index(t)

    def _observation_at_index(self, index: int) -> Dict[str, np.ndarray]:
        t = min(max(int(index), 0), self._N - 1)
        obs_out: Dict[str, np.ndarray] = {}
        for key in self._obs_keys_order:
            arr = self._obs_arrays.get(key)
            if arr is None:
                continue
            value = arr[t]
            if key == self.image_key and value.ndim >= 3:
                im_hw3 = value
                if self.crop_hw is not None:
                    ch, cw = self.crop_hw
                    H0, W0 = im_hw3.shape[:2]
                    y0 = max(0, (H0 - ch) // 2)
                    x0 = max(0, (W0 - cw) // 2)
                    im_c = im_hw3[y0:y0 + ch, x0:x0 + cw]
                    im_hw3 = cv2.resize(im_c, (W0, H0), interpolation=cv2.INTER_LINEAR)
                obs_out[key] = im_hw3
            else:
                if value.dtype != np.float32 and self._obs_modalities_map.get(key) in (None, "low_dim", "scan"):
                    value = value.astype(np.float32)
                obs_out[key] = value
        return obs_out

    def render(
        self,
        mode: str = "human",
        height: Optional[int] = None,
        width: Optional[int] = None,
        camera_name: Optional[str] = None,
        **kwargs,
    ):
        if mode == "rgb_array":
            t = min(max(self._t, 0), self._N - 1)
            if self._images is None:
                img = np.zeros((self._H or 1, self._W or 1, 3), dtype=np.uint8)
            else:
                img = self._images[t]
            if (height is not None) and (width is not None) and (height > 0) and (width > 0):
                img = cv2.resize(img, (width, height), interpolation=cv2.INTER_LINEAR)
            return img
        if mode == "human":
            img = self.render(mode="rgb_array", height=height or self._H, width=width or self._W)
            cv2.imshow(camera_name or "EnvPatcher", img[:, :, ::-1])
            cv2.waitKey(1)
            return None
        return None

    def serialize(self) -> Dict[str, Any]:
        return dict(
            env_name="Patcher",
            type=self.type,
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
                frame_stack=self.frame_stack,
                obs_keys=self._obs_keys_order,
                generated_obs_keys=sorted(self._generated_obs_keys),
                obs_key_specs={k: self._serialize_spec(v) for k, v in self._resolved_obs_specs.items()},
            ),
        )

    @staticmethod
    def _serialize_spec(spec: Dict[str, Any]) -> Dict[str, Any]:
        out = dict(spec)
        dtype = out.get("dtype")
        if isinstance(dtype, np.dtype):
            out["dtype"] = dtype.str
        return out

def create_env_patcher(dataset_path: str, *, frame_stack: int = 1, **kwargs):
    base = EnvPatcher(dataset_path=dataset_path, frame_stack=frame_stack, **kwargs)
    if frame_stack and frame_stack > 1:
        wrapped = FrameStackWrapper(base, num_frames=frame_stack)
        setattr(wrapped, "frame_stack", frame_stack)
        return wrapped
    return base
