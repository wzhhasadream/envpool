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
"""MyoSuite's success alias preserves solved and final-control-step semantics."""

from __future__ import annotations

from contextlib import ExitStack
from typing import Any

import numpy as np
from absl.testing import absltest

import envpool.mujoco.myosuite.registration  # noqa: F401
from envpool.mujoco.myosuite.tasks import MYOSUITE_TASKS
from envpool.python.glfw_context import preload_windows_gl_dlls
from envpool.registration import make_gymnasium, make_spec

preload_windows_gl_dlls(strict=True)


def _assert_alias(info: dict[str, Any], count: int) -> None:
    success, solved = info["success"], info["solved"]
    assert success.shape == solved.shape == (count,)
    assert success.dtype == solved.dtype == np.dtype(np.float64)
    assert np.all((success == 0.0) | (success == 1.0))
    np.testing.assert_array_equal(success, solved)
    assert "live" not in info and "lives" not in info


class MyoSuiteSuccessInfoTest(absltest.TestCase):
    """Keep the additive success field aligned with the native solved flag."""

    def test_all_task_and_alias_specs(self) -> None:
        """Every state/pixel registration exposes identical numeric flags."""
        self.assertLen(MYOSUITE_TASKS, 398)
        for task in MYOSUITE_TASKS:
            for task_id in (task["id"], f"MyoSuite/{task['id']}"):
                for pixels in (False, True):
                    with self.subTest(task_id=task_id, pixels=pixels):
                        spec = make_spec(task_id, from_pixels=pixels)
                        states = spec.state_array_spec
                        for key in ("info:solved", "info:success"):
                            flag = states[key]
                            self.assertEqual(flag.dtype, np.float64)
                            self.assertEqual(flag.shape, [-1])
                            self.assertEqual(flag.minimum, 0.0)
                            self.assertEqual(flag.maximum, 1.0)

    def test_mixed_slots_repeat_and_truncation(self) -> None:
        """Each slot reports its final substep and clears both flags on reset."""
        tasks = ("myoFingerReachFixed-v0", "myoFatiFingerReachRandom-v0")
        horizons = (5, 7)
        for pixels in (False, True):
            for repeat in (1, 3):
                with (
                    self.subTest(pixels=pixels, repeat=repeat),
                    ExitStack() as stack,
                ):
                    options: dict[str, Any] = dict(from_pixels=pixels)
                    if pixels:
                        options.update(render_width=32, render_height=24)
                    pool = make_gymnasium(
                        tasks[0],
                        num_envs=2,
                        seed=[23, 41],
                        env_task_names=list(tasks),
                        env_max_episode_steps=list(horizons),
                        action_repeat=repeat,
                        **options,
                    )
                    stack.callback(pool.close)
                    singles = []
                    for task, seed, horizon in zip(
                        tasks, (23, 41), horizons, strict=True
                    ):
                        env = make_gymnasium(
                            task,
                            num_envs=1,
                            seed=seed,
                            max_episode_steps=horizon,
                            **options,
                        )
                        stack.callback(env.close)
                        singles.append(env)
                    reset_info = pool.reset()[1]
                    _assert_alias(reset_info, 2)
                    np.testing.assert_array_equal(reset_info["success"], 0.0)
                    for env in singles:
                        _assert_alias(env.reset()[1], 1)
                    was_done = [False, False]
                    saw_truncation = [False, False]
                    saw_autoreset = [False, False]
                    action = np.zeros((2, 5), dtype=np.float32)
                    for _ in range(12):
                        _, _, term, trunc, info = pool.step(action)
                        _assert_alias(info, 2)
                        for slot, env in enumerate(singles):
                            for _ in range(1 if was_done[slot] else repeat):
                                expected = env.step(action[slot : slot + 1])
                                if expected[2][0] or expected[3][0]:
                                    break
                            _assert_alias(expected[4], 1)
                            self.assertEqual(term[slot], expected[2][0])
                            self.assertEqual(trunc[slot], expected[3][0])
                            self.assertEqual(
                                info["success"][slot], expected[4]["solved"][0]
                            )
                            if was_done[slot]:
                                self.assertEqual(info["success"][slot], 0.0)
                                saw_autoreset[slot] = True
                            saw_truncation[slot] |= bool(trunc[slot])
                            was_done[slot] = bool(term[slot] or trunc[slot])
                    self.assertTrue(all(saw_truncation))
                    self.assertTrue(all(saw_autoreset))

    def test_successful_step_is_cleared_by_reset(self) -> None:
        """A solved torso pose emits one, including at an early time limit."""
        for repeat in (1, 3):
            with self.subTest(repeat=repeat):
                env = make_gymnasium(
                    "myoTorsoPoseFixed-v0",
                    num_envs=1,
                    seed=5,
                    action_repeat=repeat,
                    max_episode_steps=1,
                )
                try:
                    _assert_alias(env.reset()[1], 1)
                    action = np.zeros((1, 210), dtype=np.float32)
                    _, _, term, trunc, info = env.step(action)
                    _assert_alias(info, 1)
                    self.assertEqual(info["success"][0], 1.0)
                    self.assertFalse(term[0])
                    self.assertTrue(trunc[0])
                    _, _, term, trunc, info = env.step(action)
                    _assert_alias(info, 1)
                    self.assertEqual(info["success"][0], 0.0)
                    self.assertFalse(term[0] or trunc[0])
                finally:
                    env.close()

    def test_termination_and_next_step_reset(self) -> None:
        """Dropping the pen terminates without turning failure into success."""
        for pixels in (False, True):
            for repeat in (1, 3):
                with self.subTest(pixels=pixels, repeat=repeat):
                    options: dict[str, Any] = dict(from_pixels=pixels)
                    if pixels:
                        options.update(render_width=32, render_height=24)
                    env = make_gymnasium(
                        "myoHandPenTwirlFixed-v0",
                        num_envs=1,
                        seed=5,
                        action_repeat=repeat,
                        max_episode_steps=300,
                        **options,
                    )
                    try:
                        _assert_alias(env.reset()[1], 1)
                        action = np.full((1, 39), -1.0, dtype=np.float32)
                        for _ in range(300):
                            _, _, term, trunc, info = env.step(action)
                            _assert_alias(info, 1)
                            if term[0] or trunc[0]:
                                break
                        self.assertTrue(term[0])
                        self.assertFalse(trunc[0])
                        self.assertEqual(info["success"][0], 0.0)
                        _, reward, term, trunc, info = env.step(action)
                        _assert_alias(info, 1)
                        self.assertEqual(info["success"][0], 0.0)
                        self.assertEqual(reward[0], 0.0)
                        self.assertFalse(term[0] or trunc[0])
                    finally:
                        env.close()


if __name__ == "__main__":
    absltest.main()
