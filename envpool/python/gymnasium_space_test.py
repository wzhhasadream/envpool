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
"""Exact single-space and full-vector-space contracts without native bindings."""

from types import SimpleNamespace

import gymnasium as gym
import numpy as np
from absl.testing import absltest

from envpool.python.gymnasium_envpool import GymnasiumEnvPoolMixin


class _SpacePool(GymnasiumEnvPoolMixin):
    def __init__(self, action: gym.Space, observation: gym.Space) -> None:
        self.config = {"num_envs": 5, "batch_size": 2}
        self.spec = SimpleNamespace(
            gymnasium_action_space=action,
            gymnasium_observation_space=observation,
        )


class GymnasiumSpaceTest(absltest.TestCase):
    """Batch shape, bounds, dtype, caching, and independent space state."""

    def test_discrete_action_space(self) -> None:
        """Discrete actions become one MultiDiscrete component per slot."""
        single: gym.spaces.Discrete = gym.spaces.Discrete(7, start=3)
        pool = _SpacePool(single, single)
        self.assertIs(pool.single_action_space, single)
        assert isinstance(pool.action_space, gym.spaces.MultiDiscrete)
        self.assertEqual(pool.action_space.shape, (5,))
        self.assertEqual(pool.action_space.dtype, single.dtype)
        np.testing.assert_array_equal(pool.action_space.nvec, [7] * 5)
        np.testing.assert_array_equal(pool.action_space.start, [3] * 5)
        self.assertTrue(pool.action_space.contains(pool.action_space.sample()))
        self.assertIs(pool.action_space, pool.action_space)
        self.assertIsNot(pool.action_space, pool.single_action_space)
        self.assertIs(pool.single_observation_space, single)
        self.assertIsNot(pool.observation_space, pool.single_observation_space)

    def test_box_bounds_and_dtype(self) -> None:
        """Every row preserves the exact nonuniform single-space bounds."""
        low = np.array([[-3, -2], [0, 0.25]], np.float32)
        high = np.array([[0, 1], [2, 8]], np.float32)
        single = gym.spaces.Box(low=low, high=high, dtype=np.float32)
        pool = _SpacePool(single, single)
        for space in (pool.observation_space, pool.action_space):
            assert isinstance(space, gym.spaces.Box)
            self.assertEqual(space.shape, (5, 2, 2))
            self.assertEqual(space.dtype, np.float32)
            np.testing.assert_array_equal(space.low, np.repeat(low[None], 5, 0))
            np.testing.assert_array_equal(
                space.high, np.repeat(high[None], 5, 0)
            )
        self.assertIs(pool.single_action_space, single)
        self.assertIs(pool.single_observation_space, single)
        self.assertIs(pool.observation_space, pool.observation_space)
        assert isinstance(pool.action_space, gym.spaces.Box)
        assert isinstance(pool.observation_space, gym.spaces.Box)
        pool.action_space.low[0, 0, 0] = -100
        np.testing.assert_array_equal(single.low, low)
        self.assertEqual(pool.observation_space.low[0, 0, 0], -3)

    def test_multidiscrete_action_space(self) -> None:
        """MultiDiscrete batches use Gymnasium's integer Box convention."""
        single = gym.spaces.MultiDiscrete([2, 4, 6], dtype=np.int32)
        pool = _SpacePool(single, single)
        assert isinstance(pool.action_space, gym.spaces.Box)
        self.assertEqual(pool.action_space.shape, (5, 3))
        self.assertEqual(pool.action_space.dtype, np.int32)
        np.testing.assert_array_equal(
            pool.action_space.low, np.zeros((5, 3), np.int32)
        )
        np.testing.assert_array_equal(
            pool.action_space.high, np.tile([1, 3, 5], (5, 1))
        )

    def test_nested_dict_spaces(self) -> None:
        """Nested Dict structure survives batching for observations and actions."""
        single = gym.spaces.Dict({
            "state": gym.spaces.Dict({
                "position": gym.spaces.Box(-1, 2, (3,), np.float64)
            }),
            "pixels": gym.spaces.Box(0, 255, (6, 8, 10), np.uint8),
            "choice": gym.spaces.Discrete(4),
        })
        pool = _SpacePool(single, single)
        for space in (pool.observation_space, pool.action_space):
            assert isinstance(space, gym.spaces.Dict)
            self.assertEqual(set(space.spaces), {"state", "pixels", "choice"})
            assert isinstance(space["state"], gym.spaces.Dict)
            self.assertEqual(space["state"]["position"].shape, (5, 3))
            self.assertEqual(space["pixels"].shape, (5, 6, 8, 10))
            self.assertEqual(space["pixels"].dtype, np.uint8)
            assert isinstance(space["pixels"], gym.spaces.Box)
            np.testing.assert_array_equal(
                space["pixels"].high, np.full((5, 6, 8, 10), 255, np.uint8)
            )
            assert isinstance(space["choice"], gym.spaces.MultiDiscrete)
            np.testing.assert_array_equal(space["choice"].nvec, [4] * 5)
            self.assertTrue(space.contains(space.sample()))
        self.assertIs(pool.single_observation_space, single)
        self.assertEqual(single["pixels"].shape, (6, 8, 10))

    def test_batch_sampling_does_not_advance_single_rng(self) -> None:
        """Vector samples and seeds do not alter the single action RNG stream."""
        single = gym.spaces.Box(-1, 1, (3,), np.float32, seed=19)
        reference = gym.spaces.Box(-1, 1, (3,), np.float32, seed=19)
        pool = _SpacePool(single, single)
        pool.action_space.seed(41)
        for _ in range(8):
            pool.action_space.sample()
            np.testing.assert_array_equal(single.sample(), reference.sample())


if __name__ == "__main__":
    absltest.main()
