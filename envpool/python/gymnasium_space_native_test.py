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
"""Standard Gymnasium vector spaces against separately typed native pools."""

from typing import Any

import gymnasium as gym
import numpy as np
from absl.testing import absltest, parameterized
from gymnasium.vector.utils import batch_space

import envpool.atari.registration  # noqa: F401
import envpool.mujoco.dmc.registration  # noqa: F401
import envpool.mujoco.gym.registration  # noqa: F401
import envpool.mujoco.metaworld.registration  # noqa: F401
import envpool.mujoco.myosuite.registration  # noqa: F401
from envpool.python.glfw_context import preload_windows_gl_dlls
from envpool.registration import make_dm, make_gymnasium, make_multitask

preload_windows_gl_dlls(strict=True)


def _assert_batched_space(single: Any, batched: Any, count: int) -> None:
    if isinstance(single, gym.spaces.Dict):
        assert isinstance(batched, gym.spaces.Dict)
        assert set(single.spaces) == set(batched.spaces)
        for key in single.spaces:
            _assert_batched_space(single[key], batched[key], count)
    elif isinstance(single, gym.spaces.Discrete):
        assert isinstance(batched, gym.spaces.MultiDiscrete)
        np.testing.assert_array_equal(batched.nvec, [single.n] * count)
        np.testing.assert_array_equal(batched.start, [single.start] * count)
        assert batched.dtype == single.dtype
    else:
        assert isinstance(single, gym.spaces.Box)
        assert isinstance(batched, gym.spaces.Box)
        assert batched.shape == (count, *single.shape)
        assert batched.dtype == single.dtype
        np.testing.assert_array_equal(
            batched.low, np.repeat(single.low[None], count, 0)
        )
        np.testing.assert_array_equal(
            batched.high, np.repeat(single.high[None], count, 0)
        )


class NativeGymnasiumSpaceTest(parameterized.TestCase):
    """Full-pool samples step directly; partial batches retain ID routing."""

    @parameterized.named_parameters(
        ("atari_discrete", ("Pong-v5",), {"noop_max": 1}),
        (
            "atari_continuous",
            ("Pong-v5",),
            {"continuous": True, "noop_max": 1},
        ),
        (
            "atari_multitask",
            ("Pong-v5", "Breakout-v5"),
            {"noop_max": 1},
        ),
        (
            "atari_continuous_multitask",
            ("Pong-v5", "Breakout-v5"),
            {"continuous": True, "noop_max": 1},
        ),
        ("dmc_dict", ("CartpoleBalance-v1",), {}),
        ("dmc_multitask", ("DogRun-v1", "DogStand-v1"), {}),
        ("mujoco_gym", ("HalfCheetah-v4",), {}),
        ("myosuite", ("myoFingerReachFixed-v0",), {}),
        ("metaworld", ("MetaWorld/Reach-v3",), {}),
    )
    def test_full_sync_sample_steps_directly(
        self, tasks: tuple[str, ...], options: dict[str, Any]
    ) -> None:
        """Typed pools expose original single specs and independent batched spaces."""
        if len(tasks) == 1:
            env = make_gymnasium(tasks[0], num_envs=2, num_threads=1, **options)
        else:
            env = make_multitask(tasks, num_threads=1, **options)
        try:
            self.assertEqual(env.num_envs, 2)
            self.assertEqual(
                env.single_action_space, env.spec.gymnasium_action_space
            )
            self.assertEqual(
                env.single_observation_space,
                env.spec.gymnasium_observation_space,
            )
            _assert_batched_space(env.single_action_space, env.action_space, 2)
            _assert_batched_space(
                env.single_observation_space, env.observation_space, 2
            )
            single_action, action_space = (
                env.single_action_space,
                env.action_space,
            )
            observation_space = env.observation_space
            action_space.seed(19)
            if options.get("continuous"):
                self.assertEqual(single_action.shape, (3,))
                self.assertEqual(action_space.shape, (2, 3))
                self.assertEqual(action_space.dtype, np.float32)
                np.testing.assert_array_equal(
                    single_action.low, np.array([0, -np.pi, 0], np.float32)
                )
                np.testing.assert_array_equal(
                    single_action.high, np.array([1, np.pi, 1], np.float32)
                )
            obs, info = env.reset()
            self.assertTrue(observation_space.contains(obs))
            np.testing.assert_array_equal(info["env_id"], [0, 1])
            for _ in range(8):
                action = action_space.sample()
                obs, reward, terminated, truncated, info = env.step(action)
                self.assertTrue(observation_space.contains(obs))
                for value in (reward, terminated, truncated):
                    self.assertEqual(value.shape, (2,))
                np.testing.assert_array_equal(info["env_id"], [0, 1])
            self.assertIs(env.single_action_space, single_action)
            self.assertIs(env.action_space, action_space)
            self.assertIs(env.observation_space, observation_space)
        finally:
            env.close()

    @parameterized.named_parameters(
        ("dmc", "CartpoleBalance-v1"),
        ("gym", "HalfCheetah-v4"),
        ("myosuite", "myoFingerReachFixed-v0"),
        ("metaworld", "MetaWorld/Reach-v3"),
    )
    def test_native_pixels_match_batched_box(self, task: str) -> None:
        """Native pixels unwrap to BCHW while both Box spaces retain exact bounds."""
        env = make_gymnasium(
            task,
            num_envs=2,
            num_threads=1,
            from_pixels=True,
            frame_stack=2,
            render_height=16,
            render_width=20,
        )
        try:
            self.assertEqual(env.single_observation_space.shape, (6, 16, 20))
            self.assertEqual(env.observation_space.shape, (2, 6, 16, 20))
            self.assertEqual(env.observation_space.dtype, np.uint8)
            np.testing.assert_array_equal(
                env.observation_space.low, np.zeros((2, 6, 16, 20), np.uint8)
            )
            np.testing.assert_array_equal(
                env.observation_space.high,
                np.full((2, 6, 16, 20), 255, np.uint8),
            )
            env.action_space.seed(19)
            self.assertTrue(env.observation_space.contains(env.reset()[0]))
            for _ in range(3):
                obs = env.step(env.action_space.sample())[0]
                self.assertTrue(env.observation_space.contains(obs))
        finally:
            env.close()

    @parameterized.named_parameters(
        ("discrete", "Pong-v5", {"noop_max": 1}),
        ("dict", "CartpoleBalance-v1", {}),
    )
    def test_subset_preserves_static_spaces_and_requested_ids(
        self, task: str, options: dict[str, Any]
    ) -> None:
        """Subsets follow explicit IDs without resizing full-pool space metadata."""
        env = make_gymnasium(task, num_envs=4, num_threads=1, **options)
        try:
            env.reset()
            action_space, observation_space = (
                env.action_space,
                env.observation_space,
            )
            ids = np.array([3, 1], np.int32)
            partial_action_space = batch_space(
                env.single_action_space, len(ids)
            )
            partial_observation_space = batch_space(
                env.single_observation_space, len(ids)
            )
            partial_action_space.seed(19)
            obs, info = env.reset(ids)
            self.assertTrue(partial_observation_space.contains(obs))
            np.testing.assert_array_equal(info["env_id"], ids)
            for _ in range(5):
                obs, _, _, _, info = env.step(
                    partial_action_space.sample(), ids
                )
                self.assertTrue(partial_observation_space.contains(obs))
                np.testing.assert_array_equal(info["env_id"], ids)
            self.assertIs(env.action_space, action_space)
            self.assertIs(env.observation_space, observation_space)
            _assert_batched_space(env.single_action_space, action_space, 4)
            _assert_batched_space(
                env.single_observation_space, observation_space, 4
            )
        finally:
            env.close()

    def test_async_receive_keeps_num_envs_spaces(self) -> None:
        """Async partial receives route actions using the returned environment IDs."""
        env = make_gymnasium(
            "CartpoleBalance-v1", num_envs=4, batch_size=2, num_threads=1
        )
        try:
            action_space, observation_space = (
                env.action_space,
                env.observation_space,
            )
            partial_action_space = batch_space(env.single_action_space, 2)
            partial_observation_space = batch_space(
                env.single_observation_space, 2
            )
            partial_action_space.seed(19)
            env.async_reset()
            for _ in range(8):
                obs, _, _, _, info = env.recv()
                ids = info["env_id"]
                self.assertEqual(ids.shape, (2,))
                self.assertEqual(len(set(ids)), 2)
                self.assertTrue(partial_observation_space.contains(obs))
                env.send(partial_action_space.sample(), ids)
            self.assertIs(env.action_space, action_space)
            self.assertIs(env.observation_space, observation_space)
            _assert_batched_space(env.single_action_space, action_space, 4)
            _assert_batched_space(
                env.single_observation_space, observation_space, 4
            )
        finally:
            env.close()

    def test_dm_spec_is_still_single_environment(self) -> None:
        """The Gymnasium vector change does not alter native dm_env specs."""
        env = make_dm("HalfCheetah-v4", num_envs=3, num_threads=1)
        try:
            self.assertEqual(env.action_spec().shape, (6,))
            self.assertEqual(env.observation_spec().obs.shape, (17,))
            self.assertEqual(env.reset().observation.obs.shape, (3, 17))
        finally:
            env.close()


if __name__ == "__main__":
    absltest.main()
