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
"""Native compatible-schema MyoSuite pools against independent native envs."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest import mock

import numpy as np
from absl.testing import absltest

import envpool.mujoco.myosuite.registration  # noqa: F401
from envpool.python.glfw_context import preload_windows_gl_dlls
from envpool.registration import (
    make_gymnasium,
    make_multitask,
    make_spec,
    registry,
)

preload_windows_gl_dlls(strict=True)

_TASKS = (
    "myoFingerReachFixed-v0",
    "MyoSuite/myoFatiFingerReachRandom-v0",
)
_SEEDS = (23, 41)
_HORIZONS = (5, 7)


def _assert_slot(actual: Any, expected: Any, slot: int) -> None:
    if isinstance(actual, dict):
        assert actual.keys() == expected.keys()
        for key in actual:
            if key != "env_id":
                _assert_slot(actual[key], expected[key], slot)
    else:
        np.testing.assert_array_equal(actual[slot], expected[0])


def _assert_step(actual: Any, expected: Any, slot: int) -> None:
    for left, right in zip(actual, expected, strict=True):
        _assert_slot(left, right, slot)


class MyoSuiteMultitaskTest(absltest.TestCase):
    """Keep each slot's model, metadata, seed, horizon and reset independent."""

    def test_factory_preserves_aliases_and_registry_horizons(self) -> None:
        """Compatible OSL tasks retain their distinct registered time limits."""
        tasks = (
            "myoChallengeOslRunFixed-v0",
            "MyoSuite/myoChallengeOslRunRandom-v0",
        )
        # Capture only the final pool construction. Actual native specs still
        # validate every public field, dtype, bound and pool-wide option.
        with mock.patch.object(
            registry, "make", return_value=SimpleNamespace()
        ) as constructor:
            env = make_multitask(tasks, envs_per_task=[2, 1], action_repeat=3)
        options = constructor.call_args.kwargs
        self.assertEqual(
            options["env_task_names"],
            [tasks[0], tasks[0], "myoChallengeOslRunRandom-v0"],
        )
        self.assertEqual(options["env_max_episode_steps"], [1000, 1000, 60000])
        self.assertEqual(options["num_envs"], 3)
        self.assertEqual(options["action_repeat"], 3)
        self.assertEqual(env.task_ids, tasks)
        self.assertEqual(env.env_task_ids, (tasks[0], tasks[0], tasks[1]))
        with self.assertRaisesRegex(ValueError, "Incompatible"):
            make_multitask((_TASKS[0], "myoFingerPoseFixed-v0"))

    def test_native_config_guards(self) -> None:
        """Reject unknown, incompatible or incorrectly sized slot settings."""
        for pixels in (False, True):
            for options in (
                {"env_task_names": [_TASKS[0]]},
                {"env_max_episode_steps": [5]},
                {"env_max_episode_steps": [5, 0]},
                {"env_task_names": [_TASKS[0], "not-a-myosuite-task"]},
                {"env_task_names": [_TASKS[0], "myoFingerPoseFixed-v0"]},
                {"env_task_names": [_TASKS[0], "myoHandReachFixed-v0"]},
            ):
                with self.subTest(pixels=pixels, options=options):
                    with self.assertRaises((ValueError, RuntimeError)):
                        make_spec(
                            _TASKS[0],
                            num_envs=2,
                            from_pixels=pixels,
                            **options,
                        )

    def test_mixed_slots_match_standalone_state_and_pixels(self) -> None:
        """Mixed tasks retain canonical identity and exact standalone traces."""
        for pixels in (False, True):
            with self.subTest(pixels=pixels):
                options = dict(action_repeat=3, frame_stack=2)
                if pixels:
                    options.update(
                        from_pixels=True, render_width=64, render_height=48
                    )
                pool = make_gymnasium(
                    _TASKS[0],
                    num_envs=2,
                    seed=list(_SEEDS),
                    env_task_names=list(_TASKS),
                    env_max_episode_steps=list(_HORIZONS),
                    **options,
                )
                singles = [
                    make_gymnasium(
                        task,
                        num_envs=1,
                        seed=seed,
                        max_episode_steps=horizon,
                        **options,
                    )
                    for task, seed, horizon in zip(
                        _TASKS, _SEEDS, _HORIZONS, strict=True
                    )
                ]
                try:
                    reset = pool.reset()
                    for slot, single in enumerate(singles):
                        _assert_step(reset, single.reset(), slot)
                    self.assertNotEqual(
                        reset[1]["task_id"][0], reset[1]["task_id"][1]
                    )
                    rng = np.random.default_rng(12)
                    for _ in range(12):
                        action = rng.uniform(-0.3, 0.3, size=(2, 5)).astype(
                            np.float32
                        )
                        actual = pool.step(action)
                        for slot, single in enumerate(singles):
                            _assert_step(
                                actual,
                                single.step(action[slot : slot + 1]),
                                slot,
                            )
                finally:
                    pool.close()
                    for single in singles:
                        single.close()

    def test_subset_reset_and_async_stepping_keep_slot_identity(self) -> None:
        """Only the selected slot advances or consumes a new reset sample."""
        pool = make_gymnasium(
            _TASKS[0],
            num_envs=2,
            batch_size=1,
            num_threads=2,
            seed=list(_SEEDS),
            action_repeat=3,
            env_task_names=list(_TASKS),
            env_max_episode_steps=list(_HORIZONS),
        )
        singles = [
            make_gymnasium(
                task,
                num_envs=1,
                seed=seed,
                max_episode_steps=horizon,
                action_repeat=3,
            )
            for task, seed, horizon in zip(
                _TASKS, _SEEDS, _HORIZONS, strict=True
            )
        ]
        try:
            expected_resets = [single.reset() for single in singles]
            pool.async_reset()
            for _ in range(2):
                actual = pool.recv(reset=True)
                slot = int(actual[-1]["env_id"][0])
                _assert_step(actual, expected_resets[slot], 0)
            action = np.zeros((1, 5), dtype=np.float32)
            for slot in (1, 0, 1, 1, 0, 0, 1, 0):
                actual = pool.step(action, env_id=np.array([slot], np.int32))
                self.assertEqual(int(actual[-1]["env_id"][0]), slot)
                _assert_step(actual, singles[slot].step(action), 0)
            actual_reset = pool.reset(env_id=np.array([1], np.int32))
            _assert_step(actual_reset, singles[1].reset(), 0)
            for slot in (0, 1):
                actual = pool.step(action, env_id=np.array([slot], np.int32))
                _assert_step(actual, singles[slot].step(action), 0)
        finally:
            pool.close()
            for single in singles:
                single.close()


if __name__ == "__main__":
    absltest.main()
