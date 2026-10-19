# Copyright 2026 Garena Online Private Limited
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
"""Registry, smoke, and determinism tests for native MyoSuite envs."""

from __future__ import annotations

import os
import unittest
from pathlib import Path
from typing import Any

import numpy as np
from absl.testing import absltest

import envpool.mujoco.myosuite.registration as myosuite_registration
from envpool.mujoco.myosuite.tasks import MYOSUITE_TASKS, MyoSuiteTask
from envpool.registration import list_all_envs, make_gymnasium, make_spec

_TASKS = tuple(MYOSUITE_TASKS)
_TASK_IDS = tuple(task["id"] for task in _TASKS)
_DETERMINISM_STEPS = 8
_INFO_KEYS = {
    "task_id",
    "sparse",
    "solved",
    "success",
    "oracle_numpy2_broken",
    "model_nq",
    "model_nv",
    "model_na",
}


def _shard_tasks(tasks: tuple[MyoSuiteTask, ...]) -> tuple[MyoSuiteTask, ...]:
    total_shards = int(
        os.environ.get(
            "MYOSUITE_TEST_TOTAL_SHARDS",
            os.environ.get("TEST_TOTAL_SHARDS", "1"),
        )
    )
    shard_index = int(
        os.environ.get(
            "MYOSUITE_TEST_SHARD_INDEX",
            os.environ.get("TEST_SHARD_INDEX", "0"),
        )
    )
    shard_status_file = os.environ.get("TEST_SHARD_STATUS_FILE")
    if shard_status_file:
        Path(shard_status_file).touch()
    if total_shards <= 1:
        return tasks
    if shard_index < 0 or shard_index >= total_shards:
        raise ValueError(f"invalid Bazel shard {shard_index} of {total_shards}")
    return tuple(
        task
        for index, task in enumerate(tasks)
        if index % total_shards == shard_index
    )


_SHARDED_TASKS = _shard_tasks(_TASKS)


class MyoSuiteTest(absltest.TestCase):
    """Validate registration, runtime surface, and determinism."""

    def _assert_info_equal(
        self, info0: dict[str, Any], info1: dict[str, Any]
    ) -> None:
        """Assert two vectorized EnvPool info dictionaries are equal."""
        self.assertEqual(info0.keys(), info1.keys())
        for key in info0:
            arr0 = np.asarray(info0[key])
            arr1 = np.asarray(info1[key])
            if arr0.dtype == object or arr1.dtype == object:
                self.assertEqual(arr0.shape, arr1.shape)
            else:
                np.testing.assert_array_equal(
                    arr0, arr1, err_msg=f"info[{key}]"
                )

    def test_generated_registry_matches_official_surface(self) -> None:
        """Generated metadata must cover all pinned official MyoSuite IDs."""
        self.assertLen(_TASKS, 398)
        self.assertEqual(
            tuple(myosuite_registration.myosuite_task_ids),
            _TASK_IDS,
        )
        registered = set(list_all_envs())
        for task in _TASKS:
            task_id = task["id"]
            alias = f"MyoSuite/{task_id}"
            with self.subTest(task_id=task_id):
                self.assertIn(task_id, registered)
                self.assertIn(alias, registered)
                spec = make_spec(task_id)
                alias_spec = make_spec(alias)
                self.assertEqual(
                    spec.observation_space.shape, (task["obs_dim"],)
                )
                self.assertEqual(spec.action_space.shape, (task["action_dim"],))
                self.assertEqual(
                    spec.config.max_episode_steps,
                    task["max_episode_steps"],
                )
                self.assertEqual(
                    alias_spec.observation_space.shape,
                    spec.observation_space.shape,
                )
                self.assertEqual(
                    alias_spec.action_space.shape,
                    spec.action_space.shape,
                )

    def test_reset_and_step_reference_surface(self) -> None:
        """Every registered task must reset and step with expected shapes."""
        for task in _SHARDED_TASKS:
            task_id = task["id"]
            with self.subTest(task_id=task_id):
                env = make_gymnasium(task_id, num_envs=2, seed=7)
                try:
                    obs, info = env.reset()
                    self.assertEqual(obs.shape, (2, task["obs_dim"]))
                    self.assertTrue(_INFO_KEYS.issubset(info.keys()))
                    np.testing.assert_array_equal(
                        info["success"], info["solved"]
                    )
                    action = np.zeros((2, task["action_dim"]), dtype=np.float32)
                    obs, rew, term, trunc, info = env.step(action)
                    self.assertEqual(obs.shape, (2, task["obs_dim"]))
                    self.assertEqual(rew.shape, (2,))
                    self.assertEqual(term.shape, (2,))
                    self.assertEqual(trunc.shape, (2,))
                    self.assertFalse(np.any(np.isnan(obs)))
                    self.assertFalse(np.any(np.isnan(rew)))
                    self.assertTrue(_INFO_KEYS.issubset(info.keys()))
                    np.testing.assert_array_equal(
                        info["success"], info["solved"]
                    )
                finally:
                    env.close()

    def test_no_tasks_need_oracle_numpy2_exclusion(self) -> None:
        """The pinned oracle can instantiate every official task under NumPy 2."""
        for task in _TASKS:
            self.assertFalse(task["oracle_numpy2_broken"], task["id"])

    def test_reach_reset_targets_follow_task_randomization(self) -> None:
        """Random reach targets vary reproducibly; fixed reach targets do not."""
        for task_id, random_targets in (
            ("myoFingerReachFixed-v0", False),
            ("myoFingerReachRandom-v0", True),
        ):
            with self.subTest(task_id=task_id):
                env0 = make_gymnasium(task_id, num_envs=1, seed=42)
                env1 = make_gymnasium(task_id, num_envs=1, seed=42)
                try:
                    targets = []
                    for _ in range(8):
                        obs0, info0 = env0.reset()
                        obs1, info1 = env1.reset()
                        np.testing.assert_array_equal(obs0, obs1)
                        self._assert_info_equal(info0, info1)
                        nsite = int(np.asarray(info0["model_nsite"]).item())
                        targets.append(
                            np.asarray(info0["site_pos"][0])[: 3 * nsite].copy()
                        )
                    changed = any(
                        not np.array_equal(targets[0], target)
                        for target in targets[1:]
                    )
                    self.assertEqual(changed, random_targets)
                    if random_targets:
                        sites = np.stack(targets).reshape(8, nsite, 3)
                        changed_sites = np.any(sites != sites[0], axis=(0, 2))
                        self.assertEqual(int(changed_sites.sum()), 1)
                        samples = sites[:, changed_sites, :]
                        # Pinned myoFingerReachRandom-v0 target_reach_range.
                        self.assertTrue(np.all(samples >= [0.1, -0.1, 0.1]))
                        self.assertTrue(np.all(samples <= [0.27, 0.1, 0.3]))
                finally:
                    env0.close()
                    env1.close()

    def test_reference_surface_is_deterministic(self) -> None:
        """Same seed and action sequence must produce identical rollouts."""
        rng = np.random.default_rng(123)
        for task in _SHARDED_TASKS:
            task_id = task["id"]
            with self.subTest(task_id=task_id):
                actions = rng.uniform(
                    -1.0,
                    1.0,
                    size=(_DETERMINISM_STEPS, 2, task["action_dim"]),
                ).astype(np.float32)
                env0 = make_gymnasium(task_id, num_envs=2, seed=42)
                env1 = make_gymnasium(task_id, num_envs=2, seed=42)
                try:
                    obs0, info0 = env0.reset()
                    obs1, info1 = env1.reset()
                    np.testing.assert_array_equal(obs0, obs1)
                    self._assert_info_equal(info0, info1)
                    for action in actions:
                        step0 = env0.step(action)
                        step1 = env1.step(action)
                        for value0, value1 in zip(
                            step0[:4], step1[:4], strict=True
                        ):
                            np.testing.assert_array_equal(value0, value1)
                        self._assert_info_equal(step0[4], step1[4])
                finally:
                    env0.close()
                    env1.close()


if __name__ == "__main__":
    unittest.main()
