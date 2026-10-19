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
"""One native Dog pool must reproduce its assigned standalone tasks exactly."""

from __future__ import annotations

from contextlib import ExitStack
from typing import Any

import numpy as np
from absl.testing import absltest, parameterized

import envpool.mujoco.dmc.registration  # noqa: F401
from envpool.python.glfw_context import preload_windows_gl_dlls
from envpool.registration import make_gymnasium, make_multitask, make_spec

preload_windows_gl_dlls(strict=True)

_TASKS = ("DogRun-v1", "DogStand-v1", "DogTrot-v1")
_SEEDS = (17, 41, 109)


def _row(value: Any, index: int) -> Any:
    if isinstance(value, dict):
        return {key: _row(item, index) for key, item in value.items()}
    return np.asarray(value)[index : index + 1]


def _without_env_ids(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: _without_env_ids(item)
            for key, item in value.items()
            if key != "env_id"
        }
    return value


def _assert_equal(actual: Any, expected: Any) -> None:
    if isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for key in expected:
            _assert_equal(actual[key], expected[key])
    else:
        np.testing.assert_array_equal(actual, expected)


def _assert_record_row(actual: tuple, row: int, expected: tuple) -> None:
    assert len(actual) == len(expected)
    for left, right in zip(actual[:-1], expected[:-1], strict=True):
        _assert_equal(_row(left, row), right)
    _assert_equal(
        _without_env_ids(_row(actual[-1], row)),
        _without_env_ids(expected[-1]),
    )


class DmcDogMultitaskTest(parameterized.TestCase):
    """Fixed identity, independent RNG, task horizons, and native pixels."""

    @parameterized.parameters(False, True)
    def test_matches_standalone_with_repeat_friction_and_resets(
        self, pixels: bool
    ) -> None:
        """Match all state/pixel rows and diagnostics through repeated resets."""
        options: dict[str, Any] = {
            "max_episode_steps": 7,
            "action_repeat": 3,
            "friction_multiplier": 0.35,
            "frame_stack": 2,
            "from_pixels": pixels,
        }
        if pixels:
            options.update(render_width=64, render_height=48)
        with ExitStack() as stack:
            mixed = make_multitask(
                _TASKS, seed=list(_SEEDS), num_threads=2, **options
            )
            stack.callback(mixed.close)
            refs = []
            for task, seed in zip(_TASKS, _SEEDS, strict=True):
                ref = make_gymnasium(
                    task, num_envs=1, seed=[seed], num_threads=1, **options
                )
                stack.callback(ref.close)
                refs.append(ref)
            self.assertIs(type(mixed), type(refs[0]))
            self.assertEqual(mixed.task_ids, _TASKS)
            self.assertEqual(mixed.env_task_ids, _TASKS)
            expected = [ref.reset() for ref in refs]
            actual = mixed.reset()
            for slot, record in enumerate(expected):
                _assert_record_row(actual, slot, record)
            np.testing.assert_array_equal(actual[-1]["task_id"], [1, 2, 3])
            if pixels:
                self.assertEqual(actual[0].shape, (3, 6, 48, 64))
            rng = np.random.default_rng(1109)
            for step in range(12):
                action = rng.uniform(-0.3, 0.3, (3, 38)).astype(
                    mixed.single_action_space.dtype
                )
                expected = [
                    ref.step(action[slot : slot + 1])
                    for slot, ref in enumerate(refs)
                ]
                actual = mixed.step(action)
                for slot, record in enumerate(expected):
                    _assert_record_row(actual, slot, record)
                np.testing.assert_array_equal(actual[-1]["task_id"], [1, 2, 3])
                if step in (4, 9):
                    actual_reset = mixed.reset(
                        env_id=np.array([1], dtype=np.int32)
                    )
                    _assert_record_row(actual_reset, 0, refs[1].reset())
                    np.testing.assert_array_equal(
                        actual_reset[-1]["task_id"], [2]
                    )

    def test_async_unequal_allocation_matches_standalone(self) -> None:
        """Keep row identity and RNG streams intact in a shared async pool."""
        tasks = (_TASKS[0], _TASKS[1], _TASKS[1], _TASKS[2])
        seeds = (17, 41, 67, 109)
        options: dict[str, Any] = {
            "max_episode_steps": 5,
            "action_repeat": 2,
            "friction_multiplier": 0.5,
        }
        with ExitStack() as stack:
            mixed = make_multitask(
                _TASKS,
                envs_per_task=[1, 2, 1],
                seed=list(seeds),
                num_threads=2,
                batch_size=1,
                **options,
            )
            stack.callback(mixed.close)
            self.assertTrue(mixed.is_async)
            self.assertEqual(mixed.env_task_ids, tasks)
            refs = []
            expected = []
            for task, seed in zip(tasks, seeds, strict=True):
                ref = make_gymnasium(
                    task, num_envs=1, seed=[seed], num_threads=1, **options
                )
                stack.callback(ref.close)
                ref.async_reset()
                expected.append(ref.recv())
                refs.append(ref)
            mixed.async_reset()
            rng = np.random.default_rng(211)
            for step in range(8):
                seen: set[int] = set()
                for _ in tasks:
                    record = mixed.recv()
                    slot = int(record[-1]["env_id"][0])
                    self.assertNotIn(slot, seen)
                    seen.add(slot)
                    _assert_record_row(record, 0, expected[slot])
                self.assertEqual(seen, set(range(4)))
                if step < 7:
                    action = rng.uniform(-0.3, 0.3, (4, 38)).astype(
                        mixed.single_action_space.dtype
                    )
                    expected = [
                        ref.step(action[slot : slot + 1])
                        for slot, ref in enumerate(refs)
                    ]
                    mixed.send(action)

    def test_dm_api_keeps_canonical_task_identity(self) -> None:
        """Canonical task IDs do not depend on request order or API."""
        env = make_multitask(
            ("DogTrot-v1", "DogWalk-v1", "DogRun-v1"),
            env_type="dm",
            seed=[11, 31, 51],
            action_repeat=2,
            max_episode_steps=3,
        )
        try:
            timestep = env.reset()
            np.testing.assert_array_equal(
                timestep.observation.task_id, [3, 4, 1]
            )
            action = np.zeros((3, 38))
            for _ in range(5):
                timestep = env.step(action)
                np.testing.assert_array_equal(
                    timestep.observation.task_id, [3, 4, 1]
                )
        finally:
            env.close()

    def test_rejects_incompatible_models_and_invalid_native_configs(
        self,
    ) -> None:
        """Reject incompatible backends, topology, and per-slot values early."""
        for pixels in (False, True):
            with self.assertRaises(ValueError):
                make_multitask(("DogRun-v1", "DogFetch-v1"), from_pixels=pixels)
            with self.assertRaises(ValueError):
                make_multitask(
                    ("DogRun-v1", "WalkerRun-v1"), from_pixels=pixels
                )
            for options in (
                {"env_task_names": ["run", "stand"]},
                {"env_task_names": ["run", "stand", "invalid"]},
                {"env_task_names": ["run", "stand", "fetch"]},
                {"env_max_episode_steps": [7, 11]},
                {"env_max_episode_steps": [7, 0, 5]},
                {"env_max_episode_steps": [7, -1, 5]},
            ):
                with self.subTest(pixels=pixels, options=options):
                    with self.assertRaises(ValueError):
                        make_spec(
                            "DogRun-v1",
                            num_envs=3,
                            from_pixels=pixels,
                            **options,
                        )


if __name__ == "__main__":
    absltest.main()
