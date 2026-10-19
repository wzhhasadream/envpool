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
"""Public factory contracts for the separately typed continuous Atari pools."""

import numpy as np
from absl.testing import absltest

import envpool.atari.registration  # noqa: F401
from envpool.registration import (
    EnvRegistry,
    make_dm,
    make_gymnasium,
    make_multitask,
    make_spec,
)

_TASKS = ("Pong-v5", "Breakout-v5")


class AtariContinuousApiTest(absltest.TestCase):
    """Continuous mode selects native float-action bindings without wrappers."""

    def test_spec_selects_native_continuous_action_contract(self) -> None:
        """Magnitude, angle and fire use a bounded three-component float32 action."""
        for task in _TASKS:
            with self.subTest(task=task):
                discrete = make_spec(task)
                explicit = make_spec(task, continuous=False)
                continuous = make_spec(task, continuous=True)
                self.assertFalse(discrete.config.continuous)
                self.assertEqual(discrete.config, explicit.config)
                self.assertEqual(discrete.action_space, explicit.action_space)
                self.assertTrue(continuous.config.continuous)
                self.assertEqual(
                    continuous.config.continuous_action_threshold, 0.5
                )
                space = continuous.action_space
                self.assertEqual(space.shape, (3,))
                self.assertEqual(space.dtype, np.float32)
                np.testing.assert_array_equal(
                    space.low, np.array([0, -np.pi, 0], dtype=np.float32)
                )
                np.testing.assert_array_equal(
                    space.high, np.array([1, np.pi, 1], dtype=np.float32)
                )
                self.assertEqual(
                    continuous.observation_space, discrete.observation_space
                )
                with self.assertRaisesRegex(ValueError, "action_array_spec"):
                    EnvRegistry._check_multitask_spec(
                        discrete, continuous, task
                    )

    def test_make_and_dm_select_continuous_pools(self) -> None:
        """Both public API forms instantiate the separate native continuous type."""
        gym = make_gymnasium(_TASKS[0], continuous=True, seed=31, noop_max=1)
        dm = make_dm(_TASKS[0], continuous=True, seed=31, noop_max=1)
        try:
            self.assertIn("AtariContinuous", type(gym).__name__)
            self.assertIn("AtariContinuous", type(dm).__name__)
            obs, info = gym.reset()
            timestep = dm.reset()
            np.testing.assert_array_equal(obs, timestep.observation.obs)
            action = np.array([[0.75, -0.3, 1.0]], dtype=np.float32)
            for _ in range(16):
                state = gym.step(action)
                timestep = dm.step(action)
                np.testing.assert_array_equal(
                    state[0], timestep.observation.obs
                )
                np.testing.assert_array_equal(state[1], timestep.reward)
                np.testing.assert_array_equal(
                    state[4]["task_id"], info["task_id"]
                )
        finally:
            gym.close()
            dm.close()

    def test_multitask_uses_one_continuous_native_pool(self) -> None:
        """The shared Box includes games with reduced discrete legal-action sets."""
        tasks = (*_TASKS, "Skiing-v5", "LostLuggage-v5")
        mixed = make_multitask(
            tasks,
            continuous=True,
            full_action_space=False,
            envs_per_task=[2, 1, 1, 1],
            seed=[20, 21, 22, 23, 24],
            noop_max=1,
        )
        try:
            self.assertIn("AtariContinuous", type(mixed).__name__)
            self.assertEqual(mixed.env_task_ids, (tasks[0], *tasks))
            self.assertTrue(mixed.config["continuous"])
            self.assertFalse(mixed.config["full_action_space"])
            self.assertEqual(mixed.single_action_space.shape, (3,))
            self.assertEqual(mixed.action_space.shape, (5, 3))
            _, info = mixed.reset()
            ids = np.asarray(info["task_id"]).copy()
            for _ in range(16):
                state = mixed.step(np.zeros((5, 3), dtype=np.float32))
                np.testing.assert_array_equal(state[4]["task_id"], ids)
        finally:
            mixed.close()
        with self.assertRaisesRegex(ValueError, "full_action_space"):
            make_multitask(_TASKS, full_action_space=False)

    def test_threshold_and_invalid_options(self) -> None:
        """Pass threshold through unchanged and reject malformed mode selection."""
        spec = make_spec(
            _TASKS[0], continuous=True, continuous_action_threshold=0.25
        )
        self.assertEqual(spec.config.continuous_action_threshold, 0.25)
        for invalid in ("true", 1, None):
            with (
                self.subTest(continuous=invalid),
                self.assertRaisesRegex(ValueError, "boolean"),
            ):
                make_spec(_TASKS[0], continuous=invalid)
        with self.assertRaisesRegex(ValueError, "only supported for Atari"):
            local = EnvRegistry()
            local.register(
                "other",
                "envpool.mujoco.metaworld",
                "MetaWorldEnvSpec",
                "MetaWorldDMEnvPool",
                "MetaWorldGymnasiumEnvPool",
            )
            local._resolve_spec_entry("other", False, continuous=True)


if __name__ == "__main__":
    absltest.main()
