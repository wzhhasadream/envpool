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
"""Fixed native task assignment and public multitask factory regression tests."""

from __future__ import annotations

from copy import deepcopy
from types import SimpleNamespace
from typing import Any

import numpy as np
from absl.testing import absltest

import envpool.mujoco.metaworld.registration as metaworld_registration
from envpool.python.glfw_context import preload_windows_gl_dlls
from envpool.registration import (
    EnvRegistry,
    make_gymnasium,
    make_multitask,
    make_spec,
    registry,
)

preload_windows_gl_dlls(strict=True)
_TASKS = tuple(metaworld_registration.metaworld_v3_task_ids)
_SMALL_TASKS = ("MetaWorld/Reach-v3", "MetaWorld/Push-v3")


def _compare_row(actual: Any, expected: Any, row: int) -> None:
    if isinstance(actual, dict):
        assert actual.keys() == expected.keys()
        for key in actual:
            if key != "env_id":
                _compare_row(actual[key], expected[key], row)
    elif isinstance(actual, tuple):
        for lhs, rhs in zip(actual, expected, strict=True):
            _compare_row(lhs, rhs, row)
    else:
        np.testing.assert_array_equal(
            np.asarray(actual)[row], np.asarray(expected)[0]
        )


class MetaWorldMultitaskTest(absltest.TestCase):
    """Compare mixed native slots with the existing single-task implementation."""

    def test_all_tasks_match_standalone_across_episodes(self) -> None:
        """Every task preserves physics, info, RNG and repeat/reset behavior."""
        steps = 24
        seeds = [501 + i for i in range(len(_TASKS))]
        actions = (
            np.random
            .default_rng(32)
            .uniform(-0.3, 0.3, (steps, len(_TASKS), 4))
            .astype(np.float32)
        )
        config: dict[str, Any] = dict(
            max_episode_steps=11, frame_stack=2, action_repeat=3
        )
        expected: list[list[Any]] = []
        for i, task in enumerate(_TASKS):
            single = make_gymnasium(task, num_envs=1, seed=[seeds[i]], **config)
            try:
                trajectory: list[Any] = [single.reset()]
                trajectory.extend(
                    single.step(action[i : i + 1]) for action in actions
                )
                expected.append(trajectory)
            finally:
                single.close()
                del single
        mixed = make_multitask(_TASKS, seed=seeds, **config)
        try:
            self.assertEqual(mixed.env_task_ids, _TASKS)
            self.assertEqual(mixed.task_ids, _TASKS)
            reset = mixed.reset()
            for i, trajectory in enumerate(expected):
                _compare_row(reset, trajectory[0], i)
            for step, action in enumerate(actions, 1):
                actual = mixed.step(action)
                for i, trajectory in enumerate(expected):
                    _compare_row(actual, trajectory[step], i)
        finally:
            mixed.close()

    def test_counts_subset_reset_and_canonical_ids(self) -> None:
        """Counts, noncontiguous resets and reordered slots preserve task identity."""
        mixed = make_multitask(_SMALL_TASKS, envs_per_task=[2, 3], seed=18)
        reverse = make_multitask(_SMALL_TASKS[::-1], envs_per_task=1, seed=18)
        try:
            _, info = mixed.reset()
            expected = np.asarray(info["task_id"]).copy()
            self.assertEqual(
                mixed.env_task_ids,
                (_SMALL_TASKS[0],) * 2 + (_SMALL_TASKS[1],) * 3,
            )
            _, reversed_info = reverse.reset()
            np.testing.assert_array_equal(
                reversed_info["task_id"], expected[[2, 0]]
            )
            ids = np.array([4, 0, 2], dtype=np.int32)
            _, subset = mixed.reset(ids)
            np.testing.assert_array_equal(subset["env_id"], ids)
            np.testing.assert_array_equal(subset["task_id"], expected[ids])
            step = mixed.step(np.zeros((3, 4), dtype=np.float32), ids)
            np.testing.assert_array_equal(step[4]["env_id"], ids)
            np.testing.assert_array_equal(step[4]["task_id"], expected[ids])
        finally:
            mixed.close()
            reverse.close()

    def test_async_routes_by_returned_env_id(self) -> None:
        """Mixed completion order cannot change each slot's task or RNG stream."""
        tasks = (
            _SMALL_TASKS[0],
            _SMALL_TASKS[1],
            _SMALL_TASKS[0],
            _SMALL_TASKS[1],
        )
        seeds = [93, 94, 95, 96]
        config: dict[str, Any] = dict(max_episode_steps=7, action_repeat=2)
        mixed = make_multitask(tasks, batch_size=2, seed=seeds, **config)
        singles = [
            make_gymnasium(task, seed=[seed], **config)
            for task, seed in zip(tasks, seeds, strict=True)
        ]
        actions = (
            np.random
            .default_rng(44)
            .uniform(-0.2, 0.2, (64, 4, 4))
            .astype(np.float32)
        )
        counts = np.zeros(4, dtype=np.int32)
        pending: list[Any] = [single.reset() for single in singles]
        try:
            mixed.async_reset()
            for _ in range(40):
                state = mixed.recv()
                ids = state[4]["env_id"]
                batch = []
                for row, eid in enumerate(ids):
                    if counts[eid] == 0:
                        _compare_row((state[0], state[4]), pending[eid], row)
                    else:
                        _compare_row(state, pending[eid], row)
                    action = actions[counts[eid], eid : eid + 1]
                    pending[eid] = singles[eid].step(action)
                    counts[eid] += 1
                    batch.append(action[0])
                mixed.send(np.asarray(batch), ids)
            self.assertTrue(np.all(counts > 1))
        finally:
            mixed.close()
            for single in singles:
                single.close()

    def test_pixels_repeat_and_dm_api(self) -> None:
        """Pixel shape and canonical task info remain stable through repetition."""
        env = make_multitask(
            _SMALL_TASKS,
            from_pixels=True,
            render_width=32,
            render_height=24,
            frame_stack=2,
            action_repeat=3,
            max_episode_steps=7,
            seed=[81, 82],
        )
        try:
            obs, info = env.reset()
            self.assertEqual(obs.shape, (2, 6, 24, 32))
            task_ids = np.asarray(info["task_id"]).copy()
            for _ in range(9):
                obs, _, _, _, info = env.step(
                    np.zeros((2, 4), dtype=np.float32)
                )
                self.assertEqual(obs.dtype, np.uint8)
                np.testing.assert_array_equal(info["task_id"], task_ids)
        finally:
            env.close()
        dm = make_multitask(_SMALL_TASKS, env_type="dm", seed=[81, 82])
        try:
            timestep = dm.reset()
            np.testing.assert_array_equal(
                timestep.observation.task_id, task_ids
            )
        finally:
            dm.close()

    def test_invalid_requests_and_native_mappings(self) -> None:
        """Reject ambiguous counts, overrides and invalid native task vectors."""
        for tasks, kwargs in (
            ([], {}),
            ("MetaWorld/Reach-v3", {}),
            (["missing"], {}),
            (_SMALL_TASKS, {"envs_per_task": 0}),
            (_SMALL_TASKS, {"envs_per_task": True}),
            (_SMALL_TASKS, {"envs_per_task": [1]}),
            (_SMALL_TASKS, {"envs_per_task": [1, -1]}),
            (_SMALL_TASKS, {"num_envs": 3}),
            (_SMALL_TASKS, {"task_name": "reach-v3"}),
            (_SMALL_TASKS, {"env_type": "invalid"}),
        ):
            with (
                self.subTest(tasks=tasks, kwargs=kwargs),
                self.assertRaises(ValueError),
            ):
                make_multitask(tasks, **kwargs)
        for kwargs in (
            {"env_task_names": ["reach-v3"]},
            {"env_max_episode_steps": [3]},
            {"env_max_episode_steps": [3, 0]},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                make_spec(_SMALL_TASKS[0], num_envs=2, **kwargs)
        with self.assertRaisesRegex(RuntimeError, "Unknown MetaWorld"):
            make_spec(
                _SMALL_TASKS[0],
                num_envs=2,
                env_task_names=["reach-v3", "missing-v3"],
            )

    def test_registered_task_horizons_are_preserved(self) -> None:
        """Each slot retains its own registry limit unless globally overridden."""
        local = EnvRegistry()
        for name, task_id, horizon in zip(
            ("short", "long"), _SMALL_TASKS, (3, 5), strict=True
        ):
            path, spec_cls, defaults = registry.specs[task_id]
            config = dict(defaults, max_episode_steps=horizon)
            local.register(
                name,
                path,
                spec_cls,
                "MetaWorldDMEnvPool",
                "MetaWorldGymnasiumEnvPool",
                **config,
            )
        mixed = local.make_multitask(["short", "long"], seed=[20, 21])
        singles = [
            local.make_gymnasium(name, seed=[seed])
            for name, seed in zip(("short", "long"), (20, 21), strict=True)
        ]
        try:
            reset = mixed.reset()
            for row, single in enumerate(singles):
                _compare_row(reset, single.reset(), row)
            for _ in range(17):
                action = np.zeros((2, 4), dtype=np.float32)
                state = mixed.step(action)
                for row, single in enumerate(singles):
                    _compare_row(state, single.step(action[row : row + 1]), row)
        finally:
            mixed.close()
            for single in singles:
                single.close()

    def test_complete_schema_and_config_validation(self) -> None:
        """Check dtype, bounds, shape, info fields and non-task config, not just obs size."""
        spec = make_spec(_SMALL_TASKS[0])
        for field in ("dtype", "shape", "minimum", "maximum", "is_discrete"):
            candidate: Any = SimpleNamespace(
                state_array_spec=deepcopy(spec.state_array_spec),
                action_array_spec=deepcopy(spec.action_array_spec),
            )
            value = candidate.action_array_spec["action"]
            setattr(
                value,
                field,
                {
                    "dtype": np.int32,
                    "shape": [-1, 5],
                    "minimum": -2.0,
                    "maximum": 2.0,
                    "is_discrete": True,
                }[field],
            )
            with (
                self.subTest(field=field),
                self.assertRaisesRegex(ValueError, "Incompatible"),
            ):
                EnvRegistry._check_multitask_spec(spec, candidate, "candidate")
        candidate = SimpleNamespace(
            state_array_spec=deepcopy(spec.state_array_spec),
            action_array_spec=spec.action_array_spec,
        )
        candidate.state_array_spec.pop("info:success")
        with self.assertRaisesRegex(ValueError, "keys"):
            EnvRegistry._check_multitask_spec(spec, candidate, "candidate")
        local = EnvRegistry()
        path, spec_cls, defaults = registry.specs[_SMALL_TASKS[0]]
        for name, observable in (("visible", False), ("hidden", True)):
            local.register(
                name,
                path,
                spec_cls,
                "MetaWorldDMEnvPool",
                "MetaWorldGymnasiumEnvPool",
                **defaults,
                partially_observable=observable,
            )
        with self.assertRaisesRegex(ValueError, "partially_observable"):
            local.make_multitask(["visible", "hidden"])


if __name__ == "__main__":
    absltest.main()
