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
"""All native DMC task groups reproduce their standalone task trajectories.

Legacy state specs include fields that some tasks never populate. Those fields
are explicitly excluded below, not asserted to be deterministic padding.
Actual zero-filled Dog and Quadruped padding remains part of exact comparisons.
"""

from __future__ import annotations

from contextlib import ExitStack
from copy import deepcopy
from itertools import combinations, product
from typing import Any

import numpy as np
from absl.testing import absltest, parameterized

from envpool.python.glfw_context import preload_windows_gl_dlls

# Minimal Bazel runfiles omit package-level DLL bootstrap. Load Windows GL
# before importing the native DMC module, as in the existing render tests.
preload_windows_gl_dlls(strict=True)

import envpool.mujoco.dmc.registration  # noqa: F401
from envpool.mujoco.dmc.registration import dmc_mujoco_envs
from envpool.registration import (
    make_dm,
    make_gymnasium,
    make_multitask,
    make_spec,
)
from envpool.wrappers import PixelObservationLayout, SameStepAutoReset

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


def _without_validity_masks(value: Any) -> Any:
    """Unwrap Gymnasium info masks only where the associated field exists."""
    if isinstance(value, dict):
        return {
            key: _without_validity_masks(item)
            for key, item in value.items()
            if not (key.startswith("_") and key[1:] in value)
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


# Independent expected compatibility contracts, not derived from the native
# helper or Python compatibility implementation under test. Every registered
# task must occur exactly once in each state/pixel partition.
_STATE_GROUPS = (
    ("acrobot", ("swingup", "swingup_sparse")),
    ("ball_in_cup", ("catch",)),
    ("cartpole", ("balance", "balance_sparse", "swingup", "swingup_sparse")),
    ("cartpole", ("three_poles",)),
    ("cartpole", ("two_poles",)),
    ("cheetah", ("run",)),
    ("dog", ("run", "stand", "trot", "walk")),
    ("dog", ("fetch",)),
    ("finger", ("turn_easy", "turn_hard")),
    ("finger", ("spin",)),
    ("fish", ("swim",)),
    ("fish", ("upright",)),
    ("hopper", ("hop", "stand")),
    ("humanoid", ("run", "stand", "walk")),
    ("humanoid", ("run_pure_state",)),
    ("humanoid_CMU", ("run", "stand", "walk")),
    ("lqr", ("lqr_2_1",)),
    ("lqr", ("lqr_6_2",)),
    ("manipulator", ("bring_ball", "bring_peg", "insert_ball", "insert_peg")),
    ("pendulum", ("swingup",)),
    ("point_mass", ("easy", "hard")),
    ("quadruped", ("run", "walk")),
    ("quadruped", ("escape",)),
    ("quadruped", ("fetch",)),
    ("reacher", ("easy", "hard")),
    ("stacker", ("stack_2",)),
    ("stacker", ("stack_4",)),
    ("swimmer", ("swimmer15",)),
    ("swimmer", ("swimmer6",)),
    ("walker", ("run", "stand", "walk")),
)
_PIXEL_GROUPS = tuple(
    group
    for group in _STATE_GROUPS
    if group[0] not in {"finger", "fish", "humanoid"}
) + (
    ("finger", ("spin", "turn_easy", "turn_hard")),
    ("fish", ("swim", "upright")),
    ("humanoid", ("run", "run_pure_state", "stand", "walk")),
)
_ALL_TASKS = tuple(
    (domain, task) for domain, tasks in _STATE_GROUPS for task in tasks
)
_DOMAIN_TASKS = {
    domain: tuple(sorted(task for d, task in _ALL_TASKS if d == domain))
    for domain, _ in _ALL_TASKS
}
_STATE_GROUP_ID = {
    (domain, task): index
    for index, (domain, tasks) in enumerate(_STATE_GROUPS)
    for task in tasks
}
_PIXEL_GROUP_ID = {
    (domain, task): index
    for index, (domain, tasks) in enumerate(_PIXEL_GROUPS)
    for task in tasks
}

# These names are exposed in the legacy union state spec, but WriteState does
# not assign them for the specified task. Never compare their allocator bytes.
_UNPOPULATED_OBSERVATIONS = {
    ("finger", "spin"): {"target_position", "dist_to_target"},
    ("fish", "upright"): {"target"},
    ("humanoid", "run"): {"position"},
    ("humanoid", "stand"): {"position"},
    ("humanoid", "walk"): {"position"},
    ("humanoid", "run_pure_state"): {
        "joint_angles",
        "head_height",
        "extremities",
        "torso_vertical",
        "com_velocity",
    },
}
_UNPOPULATED_INFO = {("finger", "spin"): {"target"}}


def _task_id(domain: str, task: str) -> str:
    def camel(value: str) -> str:
        return "".join(part[:1].upper() + part[1:] for part in value.split("_"))

    return f"{camel(domain)}{camel(task)}-v1"


def _canonical_id(domain: str, task: str) -> int:
    return _DOMAIN_TASKS[domain].index(task)


def _plain(value: Any) -> Any:
    """Convert DM namedtuples into dictionaries without losing any fields."""
    if hasattr(value, "_asdict"):
        return {key: _plain(item) for key, item in value._asdict().items()}
    if isinstance(value, dict):
        return {key: _plain(item) for key, item in value.items()}
    return value


def _meaningful_info(info: dict, domain: str, task: str) -> dict:
    result = {
        key: item
        for key, item in _without_env_ids(info).items()
        if key not in _UNPOPULATED_INFO.get((domain, task), set())
    }
    if domain == "manipulator" and "random_info" in result:
        # The diagnostic's last entry is written only in the uniform-object
        # spawn branch (tag 3). The other seven entries are always initialized.
        value = np.asarray(result["random_info"])
        meaningful = {"always_initialized": value[..., :7]}
        if np.all(value[..., 3] == 3):
            meaningful["uniform_object_velocity"] = value[..., 7:8]
        result["random_info"] = meaningful
    return result


def _snapshot(record: Any, domain: str, task: str) -> tuple:
    """Own every compared byte before native buffers can be recycled."""
    excluded = _UNPOPULATED_OBSERVATIONS.get((domain, task), set())
    if hasattr(record, "step_type"):
        obs_and_info = _plain(record.observation)
        obs_and_info = {
            key: item
            for key, item in obs_and_info.items()
            if key not in excluded
        }
        result = (
            record.step_type,
            record.reward,
            record.discount,
            _meaningful_info(obs_and_info, domain, task),
        )
    else:
        obs = record[0]
        if isinstance(obs, dict):
            obs = {
                key: item for key, item in obs.items() if key not in excluded
            }
        result = (
            obs,
            *record[1:-1],
            _meaningful_info(record[-1], domain, task),
        )
    return deepcopy(result)


def _env_ids(record: Any) -> np.ndarray:
    if hasattr(record, "step_type"):
        return record.observation.env_id
    return record[-1]["env_id"]


def _one_row(record: Any, row: int) -> Any:
    if hasattr(record, "step_type"):
        return record._replace(
            step_type=_row(record.step_type, row),
            reward=_row(record.reward, row),
            discount=_row(record.discount, row),
            observation=record.observation._make(
                _row(_plain(item), row) for item in record.observation
            ),
        )
    return tuple(_row(item, row) for item in record)


def _group_cases(groups: tuple) -> list[tuple]:
    return [(f"{domain}_{tasks[0]}", domain, tasks) for domain, tasks in groups]


class DmcAllTaskMultitaskTest(parameterized.TestCase):
    """Full task coverage without an oracle bridge or tolerance relaxation."""

    def _exercise_group(
        self,
        domain: str,
        tasks: tuple[str, ...],
        *,
        pixels: bool = False,
        asynchronous: bool = False,
        native_horizons: bool = True,
        dm_api: bool = False,
    ) -> None:
        # Reversed request order catches accidental request-local numeric IDs.
        requested = tuple(reversed(tasks))
        counts = (1,) * len(tasks) if pixels else (2,) + (1,) * (len(tasks) - 1)
        slots = tuple(
            task
            for task, count in zip(requested, counts, strict=True)
            for _ in range(count)
        )
        size = len(slots)
        seeds = [137 + 53 * slot for slot in range(size)]
        horizons = [
            5 + 2 * slot if native_horizons else 7 for slot in range(size)
        ]
        options: dict[str, Any] = {
            "action_repeat": 2 if pixels else 3,
            "friction_multiplier": 0.37,
            "frame_stack": 2,
            "from_pixels": pixels,
        }
        if pixels:
            options.update(render_width=40, render_height=32)
        # Enough controls for multiple horizons, next-step autoresets and two
        # explicit resets; actions vary across time, slots and actuator axes.
        steps = 16
        all_ids = tuple(range(size))
        subset = (size - 1, 0) if size > 1 else (0,)
        operations = [(True, all_ids, -1)]
        for step in range(steps):
            operations.append((False, all_ids, step))
            if step in (4, 11):
                operations.append((
                    True,
                    all_ids if asynchronous else subset,
                    step,
                ))
        expected: list[dict[int, tuple]] = [{} for _ in operations]
        actions = None
        constructor = make_dm if dm_api else make_gymnasium
        reference_type = None
        for slot, task in enumerate(slots):
            # Never keep reference renderers alive with the mixed renderers.
            ref = constructor(
                _task_id(domain, task),
                num_envs=1,
                seed=[seeds[slot]],
                num_threads=1,
                max_episode_steps=horizons[slot],
                **options,
            )
            try:
                reference_type = type(ref)
                if actions is None:
                    action_spec = ref.spec.action_array_spec["action"]
                    shape = tuple(action_spec.shape)[1:]
                    actions = (
                        np.random
                        .default_rng(7921)
                        .uniform(-0.35, 0.35, (steps, size, *shape))
                        .astype(action_spec.dtype)
                    )
                for index, (reset, ids, step) in enumerate(operations):
                    if slot not in ids:
                        continue
                    record = (
                        ref.reset()
                        if reset
                        else ref.step(actions[step, slot : slot + 1])
                    )
                    expected[index][slot] = _snapshot(record, domain, task)
            finally:
                ref.close()

        assert actions is not None
        if native_horizons:
            mixed = make_gymnasium(
                _task_id(domain, tasks[0]),
                num_envs=size,
                seed=seeds,
                num_threads=2,
                batch_size=1 if asynchronous else size,
                # Deliberately unlike every slot horizon: both the Mujoco
                # repeat loop and core final-step bookkeeping must resolve it.
                max_episode_steps=997,
                env_task_names=list(slots),
                env_max_episode_steps=horizons,
                **options,
            )
        else:
            mixed = make_multitask(
                tuple(_task_id(domain, task) for task in requested),
                env_type="dm" if dm_api else "gymnasium",
                envs_per_task=counts,
                seed=seeds,
                num_threads=2,
                batch_size=1 if asynchronous else size,
                max_episode_steps=7,
                **options,
            )
            self.assertEqual(
                mixed.task_ids,
                tuple(_task_id(domain, task) for task in requested),
            )
            self.assertEqual(
                mixed.env_task_ids,
                tuple(_task_id(domain, task) for task in slots),
            )
        try:
            self.assertIs(type(mixed), reference_type)
            self.assertEqual(mixed.is_async, asynchronous)
            self.assertEqual(tuple(mixed.config["env_task_names"]), slots)
            self.assertEqual(
                tuple(mixed.config["env_max_episode_steps"]), tuple(horizons)
            )
            truncations = np.zeros(size, dtype=int)
            for index, (reset, ids, step) in enumerate(operations):
                id_array = np.asarray(ids, dtype=np.int32)
                if asynchronous:
                    if reset:
                        mixed.async_reset()
                    else:
                        mixed.send(actions[step], id_array)
                    records = (mixed.recv(reset=reset) for _ in ids)
                else:
                    record = (
                        mixed.reset(id_array)
                        if reset
                        else mixed.step(actions[step], id_array)
                    )
                    np.testing.assert_array_equal(_env_ids(record), id_array)
                    records = (record,)
                seen: set[int] = set()
                for record in records:
                    for row, slot in enumerate(_env_ids(record)):
                        slot = int(slot)
                        self.assertNotIn(slot, seen)
                        self.assertIn(slot, ids)
                        seen.add(slot)
                        one = _one_row(record, row)
                        if not pixels:
                            observations = (
                                _plain(one.observation) if dm_api else one[0]
                            )
                            zero_keys = set()
                            if domain == "dog" and slots[slot] != "fetch":
                                zero_keys = {"ball_state", "target_position"}
                            elif domain == "quadruped":
                                if slots[slot] != "fetch":
                                    zero_keys.update((
                                        "ball_state",
                                        "target_position",
                                    ))
                                if slots[slot] != "escape":
                                    zero_keys.add("rangefinder")
                            for key in zero_keys:
                                np.testing.assert_array_equal(
                                    observations[key], 0
                                )
                        actual = _snapshot(one, domain, slots[slot])
                        _assert_record_row(actual, 0, expected[index][slot])
                        self.assertEqual(
                            int(actual[-1]["task_id"][0]),
                            _canonical_id(domain, slots[slot]),
                        )
                        if not dm_api and not reset:
                            if bool(one[3][0]):
                                truncations[slot] += 1
                                self.assertEqual(
                                    int(one[-1]["elapsed_step"][0]),
                                    horizons[slot],
                                )
                        if pixels:
                            self.assertEqual(one[0].shape, (1, 6, 32, 40))
                            self.assertEqual(one[0].dtype, np.uint8)
                self.assertEqual(seen, set(ids))
            if not dm_api and domain != "lqr":
                # LQR may finish successfully before its time limit.
                self.assertTrue(np.all(truncations >= 2), truncations)
        finally:
            mixed.close()

    def test_partitions_cover_every_registered_task_exactly_once(self) -> None:
        """Keep the independent group contract aligned with all 51 registry tasks."""
        registered = {(domain, task) for domain, task, _ in dmc_mujoco_envs}
        self.assertLen(registered, 51)
        self.assertLen(_ALL_TASKS, 51)
        self.assertEqual(set(_ALL_TASKS), registered)
        pixel_tasks = [
            (domain, task) for domain, tasks in _PIXEL_GROUPS for task in tasks
        ]
        self.assertLen(pixel_tasks, 51)
        self.assertEqual(set(pixel_tasks), registered)
        self.assertEqual(_canonical_id("swimmer", "swimmer15"), 0)
        self.assertEqual(_canonical_id("swimmer", "swimmer6"), 1)

    @parameterized.named_parameters(*[
        (
            f"{name}_{'async' if asynchronous else 'sync'}",
            domain,
            tasks,
            asynchronous,
        )
        for name, domain, tasks in _group_cases(_STATE_GROUPS)
        for asynchronous in (False, True)
    ])
    def test_native_assignment_horizons_and_exact_state_trajectory(
        self, domain: str, tasks: tuple[str, ...], asynchronous: bool
    ) -> None:
        """Compare every native slot through unequal horizons and repeated resets."""
        self._exercise_group(domain, tasks, asynchronous=asynchronous)

    @parameterized.named_parameters(*_group_cases(_STATE_GROUPS))
    def test_factory_unequal_allocation_matches_standalone(
        self, domain: str, tasks: tuple[str, ...]
    ) -> None:
        """Exercise the public factory with different counts and request order."""
        self._exercise_group(domain, tasks, native_horizons=False)

    @parameterized.named_parameters(*_group_cases(_PIXEL_GROUPS))
    def test_pixel_group_exact_trajectory_and_resets(
        self, domain: str, tasks: tuple[str, ...]
    ) -> None:
        """Match every rendered frame in each pixel-compatible task group."""
        # Separate parameterized cases let CI shard expensive render groups.
        self._exercise_group(domain, tasks, pixels=True, native_horizons=False)

    @parameterized.named_parameters(*_group_cases(_STATE_GROUPS))
    def test_dm_api_exact_trajectory_and_canonical_identity(
        self, domain: str, tasks: tuple[str, ...]
    ) -> None:
        """Match DM rewards, discounts, step types, observations and diagnostics."""
        self._exercise_group(domain, tasks, native_horizons=False, dm_api=True)

    @parameterized.parameters(False, True)
    def test_complete_native_pair_and_base_mismatch_matrix(
        self, pixels: bool
    ) -> None:
        """Validate every ordered same-domain pair and mismatched base task."""
        groups = _PIXEL_GROUP_ID if pixels else _STATE_GROUP_ID
        for domain, tasks in _DOMAIN_TASKS.items():
            for left, right in product(tasks, repeat=2):
                compatible = groups[domain, left] == groups[domain, right]
                # Also reject a base task outside the chosen slots' group.
                for assignment in ([left, right], [right, right]):
                    with self.subTest(
                        domain=domain,
                        base=left,
                        assigned=assignment,
                        pixels=pixels,
                    ):
                        options = dict(
                            num_envs=2,
                            from_pixels=pixels,
                            env_task_names=assignment,
                        )
                        if compatible:
                            spec = make_spec(_task_id(domain, left), **options)
                            self.assertEqual(
                                tuple(spec.config.env_task_names),
                                tuple(assignment),
                            )
                        else:
                            with self.assertRaises(ValueError):
                                make_spec(_task_id(domain, left), **options)

    @parameterized.parameters(False, True)
    def test_complete_incompatible_factory_pair_matrix(
        self, pixels: bool
    ) -> None:
        """Reject every incompatible task pair, including cross-domain pairs."""
        groups = _PIXEL_GROUP_ID if pixels else _STATE_GROUP_ID
        for left, right in combinations(_ALL_TASKS, 2):
            if left[0] == right[0] and groups[left] == groups[right]:
                continue
            with self.subTest(left=left, right=right, pixels=pixels):
                with self.assertRaises(ValueError):
                    make_multitask(
                        (_task_id(*left), _task_id(*right)), from_pixels=pixels
                    )

    @parameterized.parameters(False, True)
    def test_invalid_native_names_counts_and_horizons_all_domains(
        self, pixels: bool
    ) -> None:
        """Reject malformed assignment vectors before any environment starts."""
        for domain, tasks in _DOMAIN_TASKS.items():
            task = tasks[0]
            invalid_options = (
                {"task_name": "not_a_dmc_task"},
                {"env_task_names": [task]},
                {"env_task_names": [task, task, task]},
                {"env_task_names": [task, "not_a_dmc_task"]},
                {"env_max_episode_steps": [3]},
                {"env_max_episode_steps": [3, 5, 7]},
                {"env_max_episode_steps": [3, 0]},
                {"env_max_episode_steps": [-1, 5]},
            )
            for options in invalid_options:
                with self.subTest(
                    domain=domain, options=options, pixels=pixels
                ):
                    with self.assertRaises(ValueError):
                        make_spec(
                            _task_id(domain, task),
                            num_envs=2,
                            from_pixels=pixels,
                            **options,
                        )
            # Empty vectors retain ordinary standalone defaults.
            make_spec(
                _task_id(domain, task),
                num_envs=2,
                from_pixels=pixels,
                env_task_names=[],
                env_max_episode_steps=[],
            )

    def test_factory_rejects_invalid_counts_and_overrides(self) -> None:
        """Keep factory allocation and reserved configuration errors explicit."""
        tasks = ("WalkerRun-v1", "WalkerStand-v1")
        for options in (
            {"envs_per_task": [1]},
            {"envs_per_task": [1, 0]},
            {"envs_per_task": [-1, 2]},
            {"envs_per_task": [1, 1.5]},
            {"envs_per_task": True},
            {"num_envs": 3},
            {"env_task_names": ["run", "stand"]},
            {"env_max_episode_steps": [3, 5]},
            {"task_name": "walk"},
        ):
            with self.subTest(options=options):
                with self.assertRaises(ValueError):
                    make_multitask(tasks, **options)
        for invalid in ((), ("NotRegisteredDmcTask-v1",), "WalkerRun-v1"):
            with self.subTest(task_ids=invalid):
                with self.assertRaises(ValueError):
                    make_multitask(invalid)

    @parameterized.parameters(False, True)
    def test_wrapper_orders_preserve_task_and_terminal_metadata(
        self, layout_outside: bool
    ) -> None:
        """Preserve same-step final observations and identity in either layout order."""
        tasks = ("run", "stand", "walk")
        seeds = (211, 307, 401)
        horizons = (3, 5, 7)
        options = dict(
            action_repeat=2,
            friction_multiplier=0.6,
            frame_stack=2,
            from_pixels=True,
            render_width=40,
            render_height=32,
        )
        actions = np.random.default_rng(503).uniform(-0.25, 0.25, (12, 3, 6))
        initials = []
        expected: list[list[tuple[Any, Any]]] = [[] for _ in tasks]
        subset_resets = {}
        # Store and close one reference at a time, including same-step resets,
        # so wrapper coverage does not keep six render contexts alive at once.
        for slot, (task, seed, horizon) in enumerate(
            zip(tasks, seeds, horizons, strict=True)
        ):
            ref = PixelObservationLayout(
                make_gymnasium(
                    _task_id("walker", task),
                    num_envs=1,
                    seed=[seed],
                    num_threads=1,
                    max_episode_steps=horizon,
                    **options,
                )
            )
            try:
                initials.append(deepcopy(ref.reset()))
                for action in actions:
                    transition = deepcopy(ref.step(action[slot : slot + 1]))
                    ended = bool(transition[2][0] or transition[3][0])
                    current = (
                        deepcopy(ref.reset())
                        if ended
                        else (transition[0], transition[-1])
                    )
                    expected[slot].append((transition, current))
                if slot in (2, 0):
                    subset_resets[slot] = deepcopy(ref.reset())
            finally:
                ref.close()
        native = make_gymnasium(
            "WalkerRun-v1",
            num_envs=3,
            seed=list(seeds),
            num_threads=2,
            env_task_names=list(tasks),
            env_max_episode_steps=list(horizons),
            max_episode_steps=997,
            **options,
        )
        wrapped = (
            PixelObservationLayout(SameStepAutoReset(native))
            if layout_outside
            else SameStepAutoReset(PixelObservationLayout(native))
        )
        try:
            actual = wrapped.reset()
            for slot, initial in enumerate(initials):
                _assert_record_row(actual, slot, initial)
            self.assertEqual(actual[0].shape, (3, 2, 32, 40, 3))
            for step, action in enumerate(actions):
                actual = wrapped.step(action)
                done = actual[2] | actual[3]
                for slot in range(3):
                    transition, (expected_obs, expected_info) = expected[slot][
                        step
                    ]
                    for field in (1, 2, 3):
                        _assert_equal(
                            _row(actual[field], slot), transition[field]
                        )
                    if done[slot]:
                        _assert_equal(
                            actual[-1]["final_obs"][slot], transition[0][0]
                        )
                        final_info = _without_env_ids(
                            _without_validity_masks(actual[-1]["final_info"])
                        )
                        for key, value in _without_env_ids(
                            transition[-1]
                        ).items():
                            _assert_equal(_row(final_info[key], slot), value)
                        self.assertTrue(actual[-1]["_final_obs"][slot])
                        self.assertTrue(actual[-1]["_final_info"][slot])
                        self.assertTrue(
                            actual[-1]["final_info"]["_task_id"][slot]
                        )
                    elif "_final_obs" in actual[-1]:
                        self.assertFalse(actual[-1]["_final_obs"][slot])
                    _assert_equal(_row(actual[0], slot), expected_obs)
                    current_info = _without_env_ids(
                        _without_validity_masks(actual[-1])
                    )
                    for key, value in _without_env_ids(expected_info).items():
                        _assert_equal(_row(current_info[key], slot), value)
                np.testing.assert_array_equal(actual[-1]["task_id"], [0, 1, 2])
            actual = wrapped.reset(np.array([2, 0], dtype=np.int32))
            for row, slot in enumerate((2, 0)):
                _assert_record_row(actual, row, subset_resets[slot])
        finally:
            wrapped.close()


if __name__ == "__main__":
    absltest.main()
