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
"""Buffer ownership, Gymnasium info format, routing, and pixel-layout tests."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

import gymnasium as gym
import numpy as np
from absl.testing import absltest
from gymnasium.vector import AutoresetMode
from gymnasium.vector.utils import batch_space

from envpool.wrappers import PixelObservationLayout, SameStepAutoReset


def _equal(actual: Any, expected: Any) -> None:
    if isinstance(expected, dict):
        assert actual.keys() == expected.keys(), (
            actual.keys(),
            expected.keys(),
        )
        for key in expected:
            _equal(actual[key], expected[key])
    elif isinstance(expected, tuple):
        for a, b in zip(actual, expected, strict=True):
            _equal(a, b)
    elif isinstance(expected, np.ndarray) and expected.dtype == object:
        assert actual.shape == expected.shape
        for a, b in zip(actual, expected, strict=True):
            _equal(a, b)
    else:
        np.testing.assert_array_equal(actual, expected)


def _row(tree: Any, index: int) -> Any:
    if isinstance(tree, dict):
        return {key: _row(value, index) for key, value in tree.items()}
    return tree[index]


def _layout(value: Any, channels: int = 3) -> Any:
    if isinstance(value, dict):
        return {
            "pixels": _layout(value["pixels"], channels),
            "state": value["state"],
        }
    leading = value.shape[:-3]
    return (
        value
        .reshape(*leading, 2, channels, 2, 3)
        .swapaxes(-3, -2)
        .swapaxes(-2, -1)
    )


class _ReusingPool(gym.vector.VectorEnv):
    """Overwrite every receive buffer on reset, including nonselected rows."""

    def __init__(
        self, channels: int = 3, dictionary: bool = False, frames: int = 2
    ) -> None:
        self.num_envs = 4
        self.config = dict(
            num_envs=4,
            batch_size=4,
            max_num_players=1,
            from_pixels=True,
            frame_stack=frames,
        )
        if channels == 1:
            self.config = dict(
                num_envs=4,
                batch_size=4,
                max_num_players=1,
                stack_num=frames,
                gray_scale=True,
            )
        self.is_async = False
        self.dictionary = dictionary
        pixels = gym.spaces.Box(0, 255, (frames * channels, 2, 3), np.uint8)
        self.observation_space = (
            gym.spaces.Dict(
                pixels=pixels, state=gym.spaces.Box(-100, 100, (2,), np.float32)
            )
            if dictionary
            else pixels
        )
        self.single_observation_space = self.observation_space
        self.observation_space = batch_space(self.single_observation_space, 4)
        self.single_action_space = gym.spaces.Discrete(2)
        self.action_space = batch_space(self.single_action_space, 4)
        self.metadata = {
            "autoreset_mode": AutoresetMode.NEXT_STEP,
            "render_modes": ["rgb_array"],
        }
        self.render_mode = "rgb_array"
        self.spec = None
        self.task_ids = ("task_a", "task_b")
        self.env_task_ids = ("task_a", "task_b", "task_a", "task_b")
        self.obs = np.zeros((4, *pixels.shape), np.uint8)
        self.state = np.zeros((4, 2), np.float32)
        self.reward = np.zeros(4, np.float32)
        self.terminated = np.zeros(4, bool)
        self.truncated = np.zeros(4, bool)
        self.info_buffers = {
            key: np.zeros(4, np.int32)
            for key in (
                "env_id",
                "task_id",
                "success",
                "elapsed_step",
                "r",
                "l",
            )
        }
        self.calls: list[tuple[str, Any]] = []
        self.no_done = False
        self.closed = False

    def _result(self, ids: np.ndarray, reset: bool) -> Any:
        # Clobber the full original batch, not just the subset returned now.
        self.obs.fill(231)
        self.state.fill(-77)
        self.reward.fill(-99)
        self.terminated.fill(False)
        self.truncated.fill(False)
        for value in self.info_buffers.values():
            value.fill(-99)
        for row, eid in enumerate(ids):
            self.obs[row] = (
                np.arange(self.obs[row].size).reshape(self.obs.shape[1:])
                + 20 * eid
                + (0 if reset else 3)
            )
            self.state[row] = [eid, 0 if reset else 3]
            self.reward[row] = 0 if reset else 5 + eid
            self.terminated[row] = not reset and not self.no_done and eid == 0
            self.truncated[row] = not reset and not self.no_done and eid == 2
            for key, value in dict(
                env_id=eid,
                task_id=eid % 2,
                success=0 if reset else 1,
                elapsed_step=0 if reset else 3,
                r=8 + eid,
                l=3,
            ).items():
                self.info_buffers[key][row] = value
        count = len(ids)
        obs: Any = self.obs[:count]
        if self.dictionary:
            obs = {"pixels": obs, "state": self.state[:count]}
        info: dict[str, Any] = {
            key: self.info_buffers[key][:count]
            for key in ("env_id", "task_id", "success", "elapsed_step")
        }
        if reset:
            info["reset_only"] = np.full(count, 7)
        else:
            info["episode"] = {
                key: self.info_buffers[key][:count] for key in ("r", "l")
            }
        if reset:
            return obs, info
        return (
            obs,
            self.reward[:count],
            self.terminated[:count],
            self.truncated[:count],
            info,
        )

    def reset(self, env_id: Any = None, **kwargs: Any) -> Any:
        options = kwargs.get("options")
        if options and "reset_mask" in options:
            env_id = np.flatnonzero(options["reset_mask"])
        ids = np.arange(4) if env_id is None else np.asarray(env_id)
        self.calls.append(("reset", ids.copy()))
        return self._result(ids[::-1], True)

    def step(self, action: Any, env_id: Any = None) -> Any:
        if env_id is None and isinstance(action, dict):
            env_id = action.get("env_id")
        ids = np.arange(4) if env_id is None else np.asarray(env_id)
        self.calls.append(("step", ids.copy()))
        return self._result(ids[::-1], False)

    def send(self, action: Any, env_id: Any = None) -> None:
        self.pending = self.step(action, env_id)

    def recv(self, **kwargs: Any) -> Any:
        return self.pending

    def async_reset(self) -> None:
        self.pending = self.reset()

    def render(self, *args: Any, **kwargs: Any) -> Any:
        return args, kwargs

    def close(self, **kwargs: Any) -> None:
        self.closed = True


class SameStepAutoResetTest(absltest.TestCase):
    """Terminal transitions survive buffer reuse and preserve slot identity."""

    def test_gymnasium_123_info_oracle_and_buffer_reuse(self) -> None:
        """Match recursive _add_info while preserving all step buffers."""
        pool = _ReusingPool(dictionary=True)
        wrapped = SameStepAutoReset(pool)
        ids = np.array([3, 2, 0], np.int32)
        original = deepcopy(pool.step(None, ids))
        actual = wrapped.step(None, ids)
        obs, reward, terminated, truncated, info = actual
        done = original[2] | original[3]
        _equal((reward, terminated, truncated), original[1:4])
        _equal(info["env_id"], [0, 2, 3])
        _equal(pool.calls[-1], ("reset", np.array([0, 2])))
        expected_reset = deepcopy(pool.reset(np.array([0, 2])))
        oracle: Any = gym.vector.VectorEnv()
        oracle.num_envs = 3
        expected_info: dict[str, Any] = {}
        for row, eid in enumerate(original[4]["env_id"]):
            if done[row]:
                reset_row = list(expected_reset[1]["env_id"]).index(eid)
                row_info = _row(expected_reset[1], reset_row)
                row_info.update(
                    final_obs=_row(original[0], row),
                    final_info=_row(original[4], row),
                )
                _equal(_row(obs, row), _row(expected_reset[0], reset_row))
            else:
                row_info = _row(original[4], row)
                _equal(_row(obs, row), _row(original[0], row))
            expected_info = oracle._add_info(expected_info, row_info, row)
        _equal(info, expected_info)
        # The saved result must remain independent of later native reuse.
        saved = deepcopy(actual)
        pool.step(None)
        _equal(actual, saved)
        self.assertEqual(
            wrapped.metadata["autoreset_mode"], AutoresetMode.SAME_STEP
        )
        self.assertEqual(
            pool.metadata["autoreset_mode"], AutoresetMode.NEXT_STEP
        )

    def test_all_done_has_only_reset_episode_info(self) -> None:
        """An all-ended subset does not leak completed episode statistics."""
        wrapped = SameStepAutoReset(_ReusingPool())
        result = wrapped.step(None, np.array([0, 2], np.int32))
        _equal(result[4]["_final_info"], [True, True])
        self.assertNotIn("episode", result[4])
        self.assertIn("episode", result[4]["final_info"])
        _equal(result[4]["success"], [0, 0])
        _equal(result[4]["final_info"]["success"], [1, 1])

    def test_no_done_does_not_reset(self) -> None:
        """A continuing batch remains on the native fast path."""
        pool = _ReusingPool()
        pool.no_done = True
        result = SameStepAutoReset(pool).step(None)
        self.assertLen(pool.calls, 1)
        self.assertNotIn("final_info", result[4])

    def test_subset_reset_and_dictionary_action(self) -> None:
        """Subset results have subset-sized masks in returned env_id order."""
        wrapped = SameStepAutoReset(_ReusingPool())
        _equal(wrapped.reset(np.array([2, 0]))[1]["env_id"], [0, 2])
        _equal(
            wrapped.reset(options={"reset_mask": [False, True, False, True]})[
                1
            ]["env_id"],
            [3, 1],
        )
        result = wrapped.step({"env_id": np.array([3, 2]), "action": [0, 0]})
        _equal(result[4]["env_id"], [2, 3])
        _equal(result[4]["_final_info"], [True, False])
        _equal(result[4]["success"], [0, 1])
        _equal(result[4]["final_info"]["success"], [1, 0])

    def test_async_and_multiplayer_rejected_before_submission(self) -> None:
        """Unsupported entry points never leave pending native work."""
        pool = _ReusingPool()
        pool.is_async = True
        with self.assertRaisesRegex(ValueError, "synchronous"):
            SameStepAutoReset(pool)
        pool.is_async = False
        pool.config["max_num_players"] = 2
        with self.assertRaisesRegex(ValueError, "single-player"):
            SameStepAutoReset(pool)
        pool.config["max_num_players"] = 1
        wrapper = SameStepAutoReset(pool)
        for wrapped in (wrapper, PixelObservationLayout(wrapper)):
            for method in ("send", "recv", "async_reset", "xla"):
                with self.subTest(
                    method=method, wrapper=type(wrapped).__name__
                ):
                    with self.assertRaises(NotImplementedError):
                        getattr(wrapped, method)()
        self.assertEmpty(pool.calls)

    def test_invalid_subset_rejected_before_submission(self) -> None:
        """Reject duplicates and invalid IDs instead of risking a native wait."""
        pool = _ReusingPool()
        wrapper = SameStepAutoReset(pool)
        for ids in ([1, 1], [], [-1], [4], [[0]], [0.5]):
            with self.subTest(ids=ids):
                with self.assertRaises(ValueError):
                    wrapper.step(None, ids)
                with self.assertRaises(ValueError):
                    wrapper.reset(ids)
        self.assertEmpty(pool.calls)

    def test_existing_validity_masks(self) -> None:
        """Missing nested episode fields remain invalid after repacking."""
        pool: Any = _ReusingPool()
        step = pool.step

        def masked_step(*args: Any, **kwargs: Any) -> Any:
            result = step(*args, **kwargs)
            result[4]["_success"] = np.array([True, False, True, True])
            result[4]["episode"]["_r"] = np.array([False, True, True, True])
            return result

        pool.step = masked_step
        info = SameStepAutoReset(pool).step(None)[4]
        _equal(info["final_info"]["_success"], [False, False, False, True])
        _equal(info["final_info"]["episode"]["_r"], [False, True, False, True])


class PixelObservationLayoutTest(absltest.TestCase):
    """Frame/channel axes are metadata-driven and compose with autoreset."""

    def test_both_orders_dictionary_rgb_and_grayscale(self) -> None:
        """Transform both returned and final observations, keeping state info."""
        for channels in (1, 3):
            for dictionary in (False, True):
                with self.subTest(channels=channels, dictionary=dictionary):
                    first_pool = _ReusingPool(channels, dictionary)
                    second_pool = _ReusingPool(channels, dictionary)
                    first = SameStepAutoReset(
                        PixelObservationLayout(first_pool)
                    )
                    second = PixelObservationLayout(
                        SameStepAutoReset(second_pool)
                    )
                    _equal(first.reset(), second.reset())
                    actual = deepcopy(first.step(None))
                    _equal(actual, second.step(None))
                    raw: Any = _ReusingPool(channels, dictionary)
                    original = deepcopy(raw.step(None))
                    for row in (1, 3):
                        _equal(
                            actual[4]["final_obs"][row],
                            _layout(_row(original[0], row), channels),
                        )
                    space: Any = first.single_observation_space
                    if dictionary:
                        self.assertEqual(
                            space["state"],
                            raw.single_observation_space["state"],
                        )
                        space = space["pixels"]
                    self.assertEqual(space.shape, (2, 2, 3, channels))
                    self.assertEqual(
                        first.observation_space, second.observation_space
                    )
                    self.assertEqual(
                        first.metadata["autoreset_mode"],
                        AutoresetMode.SAME_STEP,
                    )
                    self.assertEqual(first.env_task_ids, raw.env_task_ids)
                    self.assertIs(first.unwrapped, first_pool)
                    self.assertEqual(
                        first.render(env_ids=[2], camera_id=1),
                        ((), {"env_ids": [2], "camera_id": 1}),
                    )

    def test_async_receive_preserves_env_id(self) -> None:
        """Pixel-only adaptation supports split-phase and partial batches."""
        pool = _ReusingPool()
        pool.is_async = True
        pool.config["batch_size"] = 2
        wrapped = PixelObservationLayout(pool)
        wrapped.async_reset()
        obs, info = wrapped.recv(reset=True)
        _equal(obs, _layout(pool.obs))
        wrapped.send(None, np.array([1, 3]))
        obs, _, _, _, info = wrapped.recv()
        _equal(info["env_id"], [3, 1])
        self.assertEqual(obs.shape, (2, 2, 2, 3, 3))
        with self.assertRaises(NotImplementedError):
            wrapped.xla()

    def test_nested_single_frame_grayscale(self) -> None:
        """A nested pixel field keeps the singleton frame and channel axes."""
        pool: Any = _ReusingPool(channels=1, frames=1)
        pool.single_observation_space = gym.spaces.Dict({
            "camera": gym.spaces.Dict({"image": pool.single_observation_space}),
            "vector": gym.spaces.Box(-100, 100, (2,), np.float32),
        })
        pool.observation_space = batch_space(pool.single_observation_space, 4)
        original_result = pool._result

        def nested_result(ids: np.ndarray, reset: bool) -> Any:
            result = original_result(ids, reset)
            obs = {
                "camera": {"image": result[0]},
                "vector": pool.state[: len(ids)],
            }
            return (obs, *result[1:])

        pool._result = nested_result
        wrapped = PixelObservationLayout(SameStepAutoReset(pool))
        obs, _ = wrapped.reset()
        self.assertEqual(obs["camera"]["image"].shape, (4, 1, 2, 3, 1))
        self.assertTrue(wrapped.observation_space.contains(obs))
        obs, _, _, _, info = wrapped.step(None)
        self.assertTrue(wrapped.observation_space.contains(obs))
        for row in np.flatnonzero(info["_final_obs"]):
            self.assertTrue(
                wrapped.single_observation_space.contains(
                    info["final_obs"][row]
                )
            )
        _equal(obs["vector"][:, 1], [3, 0, 3, 0])

    def test_space_bounds_and_metadata_validation(self) -> None:
        """Transform exact Box bounds and fail clearly on incompatible data."""
        pool = _ReusingPool()
        low = np.arange(36, dtype=np.uint8).reshape(6, 2, 3)
        pool.single_observation_space = gym.spaces.Box(
            low, low + 100, dtype=np.uint8
        )
        wrapped = PixelObservationLayout(pool)
        assert isinstance(wrapped.single_observation_space, gym.spaces.Box)
        _equal(wrapped.single_observation_space.low, _layout(low))
        _equal(wrapped.single_observation_space.high, _layout(low + 100))
        pool.config["frame_stack"] = 3
        with self.assertRaisesRegex(ValueError, "channels"):
            PixelObservationLayout(pool)
        pool.config = {}
        with self.assertRaisesRegex(ValueError, "metadata"):
            PixelObservationLayout(pool)


if __name__ == "__main__":
    absltest.main()
