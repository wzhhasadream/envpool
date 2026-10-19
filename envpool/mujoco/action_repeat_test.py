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
"""Native repeat alignment against repeated unmodified control steps.

The repeat=1 EnvPool instance is the oracle for this wrapper feature. Existing
family alignment tests remain the oracle for each individual control step.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np
from absl import logging
from absl.testing import absltest

import envpool.mujoco.dmc.registration  # noqa: F401
import envpool.mujoco.gym.registration  # noqa: F401
import envpool.mujoco.metaworld.registration  # noqa: F401
import envpool.mujoco.myosuite.registration  # noqa: F401
from envpool.python.glfw_context import preload_windows_gl_dlls
from envpool.registration import make_dm, make_gymnasium, make_spec, registry

preload_windows_gl_dlls(strict=True)

_FAMILIES = (
    "envpool.mujoco.gym",
    "envpool.mujoco.dmc",
    "envpool.mujoco.myosuite",
    "envpool.mujoco.metaworld",
)
# Aliases are covered by spec tests; run each underlying task only once.
_TASKS = tuple(
    task_id
    for task_id, (import_path, _, _) in sorted(registry.specs.items())
    if import_path in _FAMILIES
    and not task_id.startswith(("MyoSuite/", "MetaWorld/"))
)
# MetaWorld's canonical IDs carry a namespace, so include them explicitly.
_TASKS += tuple(
    task_id
    for task_id, (import_path, _, _) in sorted(registry.specs.items())
    if import_path == "envpool.mujoco.metaworld"
    and task_id.startswith("MetaWorld/")
)
_REPRESENTATIVES = (
    "HalfCheetah-v4",
    "CartpoleBalance-v1",
    "myoElbowPose1D6MRandom-v0",
    "MetaWorld/Reach-v3",
)
_TOTAL_SHARDS = int(os.environ.get("TEST_TOTAL_SHARDS", "1"))
_SHARD_INDEX = int(os.environ.get("TEST_SHARD_INDEX", "0"))


def _sharded_tasks() -> tuple[str, ...]:
    count = _TOTAL_SHARDS
    index = _SHARD_INDEX
    if not 0 <= index < count:
        raise ValueError(f"Invalid test shard {index} of {count}")
    if status_file := os.environ.get("TEST_SHARD_STATUS_FILE"):
        Path(status_file).touch()
    tasks = _TASKS[index::count]
    logging.info(
        "Task coverage: shard %d/%d contains %d of %d canonical tasks",
        index + 1,
        count,
        len(tasks),
        len(_TASKS),
    )
    return tasks


def _assert_equal(lhs: Any, rhs: Any) -> None:
    if isinstance(lhs, dict):
        assert lhs.keys() == rhs.keys()
        for key in lhs:
            _assert_equal(lhs[key], rhs[key])
    elif isinstance(lhs, tuple):
        assert len(lhs) == len(rhs)
        for left, right in zip(lhs, rhs, strict=True):
            _assert_equal(left, right)
    else:
        np.testing.assert_array_equal(lhs, rhs)


def _stack(frames: list[Any]) -> Any:
    if isinstance(frames[0], dict):
        return {key: _stack([f[key] for f in frames]) for key in frames[0]}
    return np.stack(frames, axis=1)


def _action(env: Any, rng: np.random.Generator) -> np.ndarray:
    space = env.single_action_space
    return rng.uniform(-0.3, 0.3, (1, *space.shape)).astype(space.dtype)


class ActionRepeatTest(absltest.TestCase):
    """Cover every task, inner-step limits, history and API behavior."""

    def test_registered_task_coverage(self) -> None:
        """Do not silently pass if a Bazel registration dependency is absent."""
        self.assertGreater(len(_TASKS), 500)
        self.assertEqual(
            {registry.specs[task_id][0] for task_id in _TASKS}, set(_FAMILIES)
        )

    def test_all_specs_default_and_invalid_repeats(self) -> None:
        """Validate the public configuration for state and pixel variants."""
        for task_id, (path, _, _) in registry.specs.items():
            if path not in _FAMILIES:
                continue
            for pixels in (False, True):
                with self.subTest(task_id=task_id, pixels=pixels):
                    default = make_spec(task_id, from_pixels=pixels)
                    explicit = make_spec(
                        task_id, from_pixels=pixels, action_repeat=1
                    )
                    self.assertEqual(default.config.action_repeat, 1)
                    self.assertEqual(default.config, explicit.config)
                    self.assertEqual(
                        default.observation_space, explicit.observation_space
                    )
                    self.assertEqual(
                        default.action_space, explicit.action_space
                    )
                    self.assertIn(
                        "info:per_timestep_reward", default.state_array_spec
                    )
                    for invalid in (0, -1):
                        with self.assertRaisesRegex(
                            ValueError, "action_repeat"
                        ):
                            make_spec(
                                task_id,
                                from_pixels=pixels,
                                action_repeat=invalid,
                            )
                    with self.assertRaises((
                        TypeError,
                        ValueError,
                        RuntimeError,
                    )):
                        make_spec(
                            task_id, from_pixels=pixels, action_repeat=1.5
                        )

    def test_all_tasks_repeat_one_is_default(self) -> None:
        """Explicit repeat=1 reproduces the default, including autoresets."""
        for task_id in _sharded_tasks():
            with self.subTest(task_id=task_id):
                default = make_gymnasium(
                    task_id, num_envs=1, seed=17, max_episode_steps=11
                )
                explicit = make_gymnasium(
                    task_id,
                    num_envs=1,
                    seed=17,
                    max_episode_steps=11,
                    action_repeat=1,
                )
                try:
                    _assert_equal(default.reset(), explicit.reset())
                    rng = np.random.default_rng(71)
                    for _ in range(16):
                        action = _action(default, rng)
                        _assert_equal(
                            default.step(action), explicit.step(action)
                        )
                finally:
                    default.close()
                    explicit.close()
                    del default, explicit

    def test_all_tasks_match_repeated_control_steps(self) -> None:
        """Compare summed rewards, state, diagnostics, and external stacks."""
        # 38 is intentionally not divisible by 3. Run through time limits and
        # autoresets, including stateful task histories and three-frame stacks.
        for task_id in _sharded_tasks():
            with self.subTest(task_id=task_id):
                ref = make_gymnasium(
                    task_id, num_envs=1, seed=17, max_episode_steps=38
                )
                repeated = make_gymnasium(
                    task_id,
                    num_envs=1,
                    seed=17,
                    max_episode_steps=38,
                    action_repeat=3,
                    frame_stack=3,
                )
                try:
                    obs, info = ref.reset()
                    actual, actual_info = repeated.reset()
                    frames = [obs] * 3
                    _assert_equal(actual, _stack(frames))
                    _assert_equal(info, actual_info)
                    was_done = False
                    rng = np.random.default_rng(71)
                    for _ in range(32):
                        action = _action(ref, rng)
                        rewards = []
                        # A NEXT_STEP autoreset must not consume repeated actions.
                        for _ in range(1 if was_done else 3):
                            obs, reward, term, trunc, info = ref.step(action)
                            rewards.append(reward.astype(np.float64))
                            if np.any(term | trunc):
                                break
                        (
                            actual,
                            reward,
                            actual_term,
                            actual_trunc,
                            actual_info,
                        ) = repeated.step(action)
                        frames = [obs] * 3 if was_done else [*frames[1:], obs]
                        _assert_equal(actual, _stack(frames))
                        _assert_equal(actual_term, term)
                        _assert_equal(actual_trunc, trunc)
                        _assert_equal(actual_info, info)
                        # MyoSuite/MetaWorld sum raw doubles before the public
                        # float32 cast; the reference rounds each exposed reward.
                        total = np.sum(rewards, axis=0)
                        error = np.finfo(np.float32).eps * np.maximum(
                            1.0, np.sum(np.abs(rewards), axis=0)
                        )
                        self.assertTrue(
                            np.all(np.abs(reward - total) <= error),
                            (task_id, reward, total, error),
                        )
                        was_done = bool(np.any(term | trunc))
                finally:
                    ref.close()
                    repeated.close()
                    del ref, repeated

    def test_termination_stops_before_repeat_budget(self) -> None:
        """Stop at a physical termination without crossing into a new episode."""
        ref = make_gymnasium("InvertedPendulum-v4", num_envs=1, seed=0)
        repeated = make_gymnasium(
            "InvertedPendulum-v4", num_envs=1, seed=0, action_repeat=100
        )
        try:
            ref.reset()
            repeated.reset()
            action = np.full((1, 1), 3.0, dtype=np.float64)
            rewards = []
            for _ in range(100):
                obs, reward, term, trunc, info = ref.step(action)
                rewards.append(reward)
                if np.any(term | trunc):
                    break
            self.assertTrue(term[0])
            self.assertLess(len(rewards), 100)
            actual, reward, actual_term, actual_trunc, actual_info = (
                repeated.step(action)
            )
            _assert_equal(actual, obs)
            _assert_equal(actual_info, info)
            _assert_equal(actual_term, term)
            _assert_equal(actual_trunc, trunc)
            np.testing.assert_array_equal(reward, np.sum(rewards, axis=0))
        finally:
            ref.close()
            repeated.close()

    def test_dm_discount_and_partial_repeat(self) -> None:
        """Keep each family's existing discount convention and reset behavior."""
        for task_id in _REPRESENTATIVES:
            with self.subTest(task_id=task_id):
                ref = make_dm(task_id, num_envs=1, seed=17, max_episode_steps=5)
                repeated = make_dm(
                    task_id,
                    num_envs=1,
                    seed=17,
                    max_episode_steps=5,
                    action_repeat=3,
                )
                try:
                    ref.reset()
                    repeated.reset()
                    spec = make_spec(task_id)
                    action = np.zeros(
                        (1, *spec.action_space.shape), spec.action_space.dtype
                    )
                    for repeats in (3, 2, 1):
                        for _ in range(repeats):
                            expected = ref.step(action)
                            if expected.last()[0]:
                                break
                        actual = repeated.step(action)
                        if (
                            actual.first()[0]
                            and task_id == "CartpoleBalance-v1"
                        ):
                            np.testing.assert_array_equal(actual.reward, 0.0)
                        np.testing.assert_array_equal(
                            actual.step_type, expected.step_type
                        )
                        np.testing.assert_array_equal(
                            actual.discount, expected.discount
                        )
                        _assert_equal(actual.observation, expected.observation)
                finally:
                    ref.close()
                    repeated.close()
                    del ref, repeated

    def test_pixels_publish_and_stack_only_external_frames(self) -> None:
        """Pixel frames match manual repetition and stack once per call."""
        for task_id in (
            *_REPRESENTATIVES,
            "QuadrupedWalk-v1",
            "CheetahRun-v1",
            "FingerSpin-v1",
        ):
            with self.subTest(task_id=task_id):
                options = dict(
                    num_envs=1,
                    seed=17,
                    max_episode_steps=8,
                    from_pixels=True,
                    render_mode="rgb_array",
                    render_width=64,
                    render_height=48,
                )
                ref = make_gymnasium(task_id, **options)
                repeated = make_gymnasium(
                    task_id, **options, action_repeat=3, frame_stack=3
                )
                try:
                    obs, info = ref.reset()
                    actual, actual_info = repeated.reset()
                    frames = [obs] * 3
                    _assert_equal(actual, np.concatenate(frames, axis=1))
                    _assert_equal(info, actual_info)
                    was_done = False
                    rng = np.random.default_rng(71)
                    for _ in range(8):
                        action = _action(ref, rng)
                        for _ in range(1 if was_done else 3):
                            obs, _, term, trunc, info = ref.step(action)
                            if np.any(term | trunc):
                                break
                        actual, _, actual_term, actual_trunc, actual_info = (
                            repeated.step(action)
                        )
                        frames = [obs] * 3 if was_done else [*frames[1:], obs]
                        _assert_equal(actual, np.concatenate(frames, axis=1))
                        rendered = repeated.render(env_ids=[0])
                        assert rendered is not None
                        _assert_equal(
                            actual[:, -3:], np.transpose(rendered, (0, 3, 1, 2))
                        )
                        _assert_equal(actual_info, info)
                        _assert_equal(actual_term, term)
                        _assert_equal(actual_trunc, trunc)
                        was_done = bool(np.any(term | trunc))
                finally:
                    ref.close()
                    repeated.close()
                    del ref, repeated

    def test_async_envs_have_independent_repeat_budgets(self) -> None:
        """Selective asynchronous stepping cannot advance another env's clock."""
        env = make_gymnasium(
            "HalfCheetah-v4",
            num_envs=2,
            batch_size=1,
            num_threads=2,
            seed=17,
            max_episode_steps=5,
            action_repeat=3,
        )
        try:
            env.async_reset()
            first = env.recv(reset=True)
            second = env.recv(reset=True)
            self.assertEqual(
                {int(first[1]["env_id"][0]), int(second[-1]["env_id"][0])},
                {0, 1},
            )
            action = np.zeros((1, 6), np.float64)
            for env_id, elapsed, truncated in (
                (0, 3, False),
                (0, 5, True),
                (1, 3, False),
                (0, 0, False),
                (1, 5, True),
                (1, 0, False),
            ):
                _, _, term, trunc, info = env.step(
                    action, env_id=np.array([env_id], np.int32)
                )
                self.assertEqual(info["env_id"][0], env_id)
                self.assertEqual(info["elapsed_step"][0], elapsed)
                self.assertEqual(trunc[0], truncated)
                self.assertFalse(term[0])
        finally:
            env.close()


if __name__ == "__main__":
    # Shard tasks inside each all-task test, rather than sharding test methods
    # as well (which would silently omit most tasks from each method).
    os.environ.pop("TEST_TOTAL_SHARDS", None)
    os.environ.pop("TEST_SHARD_INDEX", None)
    absltest.main()
