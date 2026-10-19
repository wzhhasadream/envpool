# Copyright 2026 EnvPool Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Opt-in Gymnasium adapters; native EnvPool defaults are unchanged."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium.vector.utils import batch_space

__all__ = ["SameStepAutoReset", "PixelObservationLayout"]


def _tree_row(value: Any, index: int) -> Any:
    if isinstance(value, dict):
        return {key: _tree_row(child, index) for key, child in value.items()}
    return value[index].copy()


def _replace_rows(target: Any, source: Any, rows: np.ndarray) -> None:
    if isinstance(target, dict):
        for key in target:
            _replace_rows(target[key], source[key], rows)
    else:
        target[rows] = source


def _scatter_info(
    target: dict[str, Any],
    source: dict[str, Any],
    rows: np.ndarray,
    size: int,
    valid: np.ndarray,
) -> None:
    """Pack one batch into Gymnasium's recursive info-and-validity format."""
    for key, value in source.items():
        if key.startswith("_") and key[1:] in source:
            continue  # Existing validity masks describe their associated key.
        present = valid & np.asarray(source.get(f"_{key}", True), dtype=bool)
        if not np.any(present):
            continue
        if isinstance(value, dict):
            _scatter_info(
                target.setdefault(key, {}), value, rows, size, present
            )
        else:
            value = np.asarray(value)
            if value.ndim == 0:
                value = np.broadcast_to(value, (len(rows),))
            if value.shape[0] != len(rows):
                raise ValueError(
                    f"Info field {key!r} is not batched by env_id."
                )
            if key not in target:
                target[key] = np.zeros((size, *value.shape[1:]), value.dtype)
            target[key][rows[present]] = value[present]
        mask = target.setdefault(f"_{key}", np.zeros(size, dtype=bool))
        mask[rows[present]] = True


class _EnvPoolWrapper(gym.vector.VectorEnv):
    """Forward the native Gymnasium EnvPool API and public attributes."""

    # A constructor that fails must not close the caller's environment.
    closed = True

    def __init__(self, env: Any) -> None:
        if not isinstance(env, gym.vector.VectorEnv) or not hasattr(
            env, "config"
        ):
            raise TypeError("These wrappers require a Gymnasium EnvPool.")
        self.env: Any = env
        self.num_envs = env.num_envs
        self.observation_space = env.observation_space
        self.single_observation_space = env.single_observation_space
        self.action_space = env.action_space
        self.single_action_space = env.single_action_space
        self.metadata = dict(env.metadata)
        self.render_mode = env.render_mode
        self.spec = env.spec
        self.closed = False

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            raise AttributeError(name)
        return getattr(self.env, name)

    def __len__(self) -> int:
        return self.num_envs

    @property
    def unwrapped(self) -> Any:
        """The native environment beneath all wrappers."""
        return self.env.unwrapped

    def reset(self, env_id: Any = None, **kwargs: Any) -> Any:
        """Forward full or subset reset, including Gymnasium reset options."""
        return self.env.reset(env_id, **kwargs)

    def render(self, *args: Any, **kwargs: Any) -> Any:
        """Forward render unchanged, including env_ids and camera_id."""
        return self.env.render(*args, **kwargs)

    def close(self, **kwargs: Any) -> None:
        """Close the wrapped pool."""
        self.env.close(**kwargs)
        self.closed = True

    def xla(self, *args: Any, **kwargs: Any) -> Any:
        """Reject compiled calls that would bypass this Python adapter."""
        raise NotImplementedError(
            "Optional Python wrappers do not support XLA."
        )


class SameStepAutoReset(_EnvPoolWrapper):
    """Reset ended slots before returning a synchronous Gymnasium step.

    Rewards and done flags belong to the completed transition. Its observation
    is in ``info['final_obs']`` (an object array), and its information is in
    ``info['final_info']`` (a nested dictionary of batched arrays). Both have
    Gymnasium validity masks. Main observations and info belong to the reset
    episode for ended slots, and to the current transition for all other slots.

    Only single-player synchronous pools are supported. Use ``step``/``reset``;
    split-phase send/recv, async_reset, and XLA are explicitly unsupported.
    Subsets keep the order of the underlying result's ``info['env_id']``.
    """

    def __init__(self, env: Any) -> None:
        """Wrap a NEXT_STEP pool without changing its native configuration."""
        try:
            from gymnasium.vector import AutoresetMode
        except ImportError as exc:
            raise ImportError(
                "SameStepAutoReset requires Gymnasium >= 1.1."
            ) from exc
        if env.is_async or env.config.get("max_num_players", 1) != 1:
            raise ValueError(
                "SameStepAutoReset requires a synchronous single-player pool."
            )
        if env.metadata.get("autoreset_mode") != AutoresetMode.NEXT_STEP:
            raise ValueError("SameStepAutoReset requires NEXT_STEP autoreset.")
        super().__init__(env)
        self.metadata["autoreset_mode"] = AutoresetMode.SAME_STEP

    def _env_ids(self, ids: Any) -> np.ndarray:
        result = np.asarray(ids)
        if (
            result.ndim != 1
            or result.size == 0
            or not np.issubdtype(result.dtype, np.integer)
            or np.any(result < 0)
            or np.any(result >= self.num_envs)
            or len(np.unique(result)) != len(result)
        ):
            raise ValueError(
                "env_id must be a nonempty vector of unique valid slot IDs."
            )
        return result.astype(np.int32, copy=False)

    def reset(self, env_id: Any = None, **kwargs: Any) -> Any:
        """Reset all or selected slots; return only the selected batch."""
        if env_id is not None:
            env_id = self._env_ids(env_id)
        return self.env.reset(env_id, **kwargs)

    def step(self, action: Any, env_id: Any = None) -> Any:
        """Step once, then reset precisely the returned ended slot IDs."""
        if env_id is not None:
            env_id = self._env_ids(env_id)
        elif isinstance(action, dict) and "env_id" in action:
            action = {**action, "env_id": self._env_ids(action["env_id"])}
        result = self.env.step(action, env_id)
        obs, reward, terminated, truncated, info = result
        done = np.asarray(terminated) | np.asarray(truncated)
        if not np.any(done):
            return result

        # Reset may reuse ANY native receive buffer, including live rows and
        # rewards. Own the entire transition before asking native code to reset.
        obs, reward, terminated, truncated, info = deepcopy(result)
        done = np.asarray(terminated) | np.asarray(truncated)
        ids = self._env_ids(info["env_id"])
        if done.shape != ids.shape:
            raise ValueError(
                "Done flags must have one entry per returned env_id."
            )
        size = len(ids)
        final_obs = np.full(size, None, dtype=object)
        for row in np.flatnonzero(done):
            final_obs[row] = _tree_row(obs, int(row))
        final_info: dict[str, Any] = {}
        rows = np.arange(size)
        _scatter_info(final_info, info, rows, size, done)

        reset_obs, reset_info = self.env.reset(ids[done])
        reset_ids = self._env_ids(reset_info["env_id"])
        if set(reset_ids) != set(ids[done]):
            raise RuntimeError(
                "Subset reset returned unexpected env_id values."
            )
        positions = {int(eid): row for row, eid in enumerate(ids)}
        reset_rows = np.array([positions[int(eid)] for eid in reset_ids])
        _replace_rows(obs, reset_obs, reset_rows)
        current_info: dict[str, Any] = {}
        _scatter_info(current_info, info, rows, size, ~done)
        _scatter_info(
            current_info,
            reset_info,
            reset_rows,
            size,
            np.ones(len(reset_rows), dtype=bool),
        )
        current_info.update(
            final_obs=final_obs,
            _final_obs=done.copy(),
            final_info=final_info,
            _final_info=done.copy(),
        )
        return obs, reward, terminated, truncated, current_info

    def send(self, *args: Any, **kwargs: Any) -> None:
        """Reject split-phase submission before it can modify the pool."""
        raise NotImplementedError(
            "SameStepAutoReset supports synchronous step/reset only."
        )

    def recv(self, *args: Any, **kwargs: Any) -> Any:
        """Reject split-phase receive instead of silently bypassing autoreset."""
        raise NotImplementedError(
            "SameStepAutoReset supports synchronous step/reset only."
        )

    def async_reset(self, *args: Any, **kwargs: Any) -> None:
        """Reject asynchronous reset before submitting native work."""
        raise NotImplementedError(
            "SameStepAutoReset supports synchronous step/reset only."
        )


class PixelObservationLayout(_EnvPoolWrapper):
    """Expose native stacked pixels as ``[B, F, H, W, C]`` without resizing.

    Frame/channel counts come from MuJoCo's ``frame_stack`` or Atari's
    ``stack_num``/``gray_scale`` configuration. Grayscale keeps ``C=1``.
    Single-environment spaces become ``[F, H, W, C]``. Dictionary pixel fields
    are transformed recursively; nonpixel fields and rendering are unchanged.
    Supports synchronous and split-phase APIs and either SameStepAutoReset order.
    """

    def __init__(self, env: Any) -> None:
        """Validate native pixel metadata and construct transformed spaces."""
        config = env.config
        if "stack_num" in config and "gray_scale" in config:
            self.frames = int(config["stack_num"])
            self.channels = 1 if config["gray_scale"] else 3
        elif "frame_stack" in config:
            self.frames = int(config["frame_stack"])
            self.channels = 3
        else:
            raise ValueError(
                "PixelObservationLayout requires native pixel stack metadata."
            )
        if self.frames < 1:
            raise ValueError("The native frame count must be positive.")
        self._pixel_paths: set[tuple[str, ...]] = set()
        space = self._space(env.single_observation_space, ())
        if not self._pixel_paths:
            raise ValueError("No native uint8 pixel observations were found.")
        super().__init__(env)
        self.single_observation_space = space
        self.observation_space = batch_space(space, self.num_envs)

    def _pixels(self, value: np.ndarray) -> np.ndarray:
        leading = value.shape[:-3]
        height, width = value.shape[-2:]
        return np.moveaxis(
            value.reshape(*leading, self.frames, self.channels, height, width),
            -3,
            -1,
        )

    def _space(self, space: gym.Space, path: tuple[str, ...]) -> gym.Space:
        if isinstance(space, gym.spaces.Dict):
            return gym.spaces.Dict({
                key: self._space(child, (*path, key))
                for key, child in space.spaces.items()
            })
        if (
            isinstance(space, gym.spaces.Box)
            and space.dtype == np.uint8
            and len(space.shape) == 3
        ):
            if space.shape[0] != self.frames * self.channels:
                raise ValueError(
                    "Pixel space channels do not match native stack metadata."
                )
            self._pixel_paths.add(path)
            return gym.spaces.Box(
                self._pixels(space.low),
                self._pixels(space.high),
                dtype=np.uint8,
            )
        return space

    def _observation(self, value: Any, path: tuple[str, ...] = ()) -> Any:
        if path in self._pixel_paths:
            return self._pixels(value)
        if isinstance(value, dict):
            return {
                key: self._observation(child, (*path, key))
                for key, child in value.items()
            }
        return value

    def _step_result(self, result: Any) -> Any:
        obs, reward, terminated, truncated, info = result
        if "final_obs" in info:
            info = dict(info)
            final_obs = info["final_obs"].copy()
            for row in np.flatnonzero(info["_final_obs"]):
                final_obs[row] = self._observation(final_obs[row])
            info["final_obs"] = final_obs
        return self._observation(obs), reward, terminated, truncated, info

    def reset(self, env_id: Any = None, **kwargs: Any) -> Any:
        """Transform full or subset reset observations."""
        obs, info = self.env.reset(env_id, **kwargs)
        return self._observation(obs), info

    def step(self, action: Any, env_id: Any = None) -> Any:
        """Transform observations and any masked terminal observations."""
        return self._step_result(self.env.step(action, env_id))

    def recv(self, reset: bool = False, return_info: bool = True) -> Any:
        """Transform split-phase receives without changing their env_id routing."""
        result = self.env.recv(reset=reset, return_info=return_info)
        if reset:
            obs, info = result
            return self._observation(obs), info
        return self._step_result(result)
