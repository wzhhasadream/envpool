# Copyright 2026 Garena Online Private Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Native continuous Atari batching, wrapper, and invalid-input regression tests."""

from typing import Any

import numpy as np
from absl.testing import absltest, parameterized
from envpool.atari.atari_envpool import _continuous_action

import envpool.atari.registration as reg
from envpool.registration import (
    make_dm,
    make_gymnasium,
    make_multitask,
    make_spec,
)


class AtariContinuousTest(parameterized.TestCase):
    """Continuous actions use the same native state and worker infrastructure."""

    @parameterized.named_parameters(
        ("rgb", dict(gray_scale=False, stack_num=2, frame_skip=1)),
        (
            "life_and_sticky",
            dict(
                episodic_life=True,
                zero_discount_on_life_loss=True,
                reward_clip=True,
                frame_skip=4,
                repeat_action_probability=0.25,
            ),
        ),
    )
    def test_mixed_wrappers_reset_and_render_match_standalone(
        self, options: dict[str, Any]
    ) -> None:
        """Fractional paddle strengths and each wrapper remain slot-local."""
        tasks = ("Breakout-v5", "Skiing-v5", "LostLuggage-v5")
        slots = (tasks[0], tasks[1], tasks[1], tasks[2])
        seeds = [101, 203, 305, 407]
        kwargs: dict[str, Any] = dict(
            continuous=True,
            full_action_space=False,
            max_episode_steps=83,
            render_mode="rgb_array",
            **options,
        )
        mixed = make_multitask(
            tasks, envs_per_task=[1, 2, 1], seed=seeds, **kwargs
        )
        singles = [
            make_gymnasium(task, seed=seed, num_threads=1, **kwargs)
            for task, seed in zip(slots, seeds, strict=True)
        ]
        try:
            self._compare_batch(
                mixed.reset(), [single.reset() for single in singles]
            )
            rng = np.random.default_rng(52)
            for step in range(512):
                actions = rng.uniform(
                    [0, -np.pi, 0], [1, np.pi, 1], (4, 3)
                ).astype(np.float32)
                self._compare_batch(
                    mixed.step(actions),
                    [
                        single.step(actions[i : i + 1])
                        for i, single in enumerate(singles)
                    ],
                )
                if step % 71 == 0:
                    ids = np.array([3, 1], np.int32)
                    self._compare_batch(
                        mixed.reset(ids),
                        [singles[slot].reset() for slot in ids],
                    )
                if step % 64 == 0:
                    frames = mixed.render(env_ids=[2, 0, 3])
                    assert frames is not None
                    for row, slot in enumerate([2, 0, 3]):
                        reference = singles[slot].render()
                        assert reference is not None
                        np.testing.assert_array_equal(frames[row], reference[0])
        finally:
            mixed.close()
            for single in singles:
                single.close()

    def _compare_batch(self, actual: tuple, expected: list[tuple]) -> None:
        for row, single in enumerate(expected):
            for left, right in zip(actual[:-1], single[:-1], strict=True):
                np.testing.assert_array_equal(left[row], right[0])
            for key in (
                "task_id",
                "lives",
                "reward",
                "terminated",
                "ram",
                "elapsed_step",
            ):
                np.testing.assert_array_equal(
                    actual[-1][key][row], single[-1][key][0]
                )

    def test_async_batches_with_continuous_actions(self) -> None:
        """Asynchronous batches retain their assigned game and paddle state."""
        tasks = ("Breakout-v5", "Skiing-v5", "LostLuggage-v5")
        kwargs: dict[str, Any] = dict(
            continuous=True,
            repeat_action_probability=0.25,
            max_episode_steps=43,
        )
        mixed = make_multitask(
            tasks,
            envs_per_task=2,
            seed=[10, 11, 20, 21, 30, 31],
            batch_size=2,
            num_threads=3,
            **kwargs,
        )
        slots = tuple(task for task in tasks for _ in range(2))
        singles = [
            make_gymnasium(task, seed=seed, num_threads=1, **kwargs)
            for task, seed in zip(slots, [10, 11, 20, 21, 30, 31], strict=True)
        ]
        try:
            pending = []
            for single in singles:
                obs, info = single.reset()
                pending.append((
                    obs,
                    np.zeros(1, np.float32),
                    np.zeros(1, bool),
                    np.zeros(1, bool),
                    info,
                ))
            mixed.async_reset()
            counts = np.zeros(6, np.int32)
            for _ in range(256):
                actual = mixed.recv()
                ids = actual[-1]["env_id"]
                self._compare_batch(actual, [pending[slot] for slot in ids])
                actions = np.empty((2, 3), np.float32)
                for row, slot in enumerate(ids):
                    counts[slot] += 1
                    actions[row] = [
                        (counts[slot] % 11) / 10,
                        np.sin(counts[slot]),
                        (counts[slot] + slot) % 2,
                    ]
                    pending[slot] = singles[slot].step(actions[row : row + 1])
                mixed.send(actions, env_id=ids)
            self.assertTrue(np.all(counts > 0), counts)
        finally:
            mixed.close()
            for single in singles:
                single.close()

    @parameterized.parameters(
        [np.nan, 0, 0],
        [0, np.inf, 0],
        [0, 0, -np.inf],
        [-0.01, 0, 0],
        [1.01, 0, 0],
        [0, -4, 0],
        [0, 4, 0],
        [0, 0, -0.01],
        [0, 0, 1.01],
    )
    def test_invalid_batch_rejected_before_any_slot_steps(
        self, radius: float, theta: float, fire: float
    ) -> None:
        """Invalid float inputs fail synchronously without partially stepping."""
        kwargs: dict[str, Any] = dict(
            continuous=True, seed=[74, 83], num_threads=1, num_envs=2
        )
        actual = make_gymnasium("Breakout-v5", **kwargs)
        reference = make_gymnasium("Breakout-v5", **kwargs)
        try:
            actual.reset()
            reference.reset()
            actions = np.array([[1, 0, 1], [radius, theta, fire]], np.float32)
            # Validate after a successful send too: this cannot rely on the
            # generic Python wrapper's one-time dtype/shape check.
            valid = np.ones((2, 3), np.float32)
            actual.step(valid)
            reference.step(valid)
            with self.assertRaises(ValueError):
                actual.send(actions)
            left, right = actual.step(valid), reference.step(valid)
            for a, b in zip(left[:-1], right[:-1], strict=True):
                np.testing.assert_array_equal(a, b)
            for key in ("ram", "elapsed_step", "lives"):
                np.testing.assert_array_equal(left[-1][key], right[-1][key])
        finally:
            actual.close()
            reference.close()

    def test_pinned_threshold_quirks_and_float_boundaries(self) -> None:
        """The exact ALE 0.11.2 rule is intentionally not a symmetric dead zone."""
        cases = [
            ([0, 0, 0], 9),
            ([1, 0, 0], 8),
            ([0.5, 0, 0.5], 5),
            ([1, np.pi / 2, 1], 15),
            ([0.5, 0, np.nextafter(np.float32(0.5), np.float32(1))], 13),
        ]
        for values, expected in cases:
            action = np.asarray(values, np.float32)
            self.assertEqual(
                _continuous_action(action, 0.5), (expected, float(action[0]))
            )

    def test_all_104_games_share_one_continuous_pool(self) -> None:
        """The complete catalog can coexist, including both reduced legal sets."""
        tasks = [
            "".join(part.capitalize() for part in game.split("_")) + "-v5"
            for game in reg.atari_game_list
        ]
        self.assertLen(tasks, 104)
        env = make_multitask(
            tasks,
            continuous=True,
            num_threads=4,
            noop_max=1,
            use_fire_reset=False,
            frame_skip=1,
        )
        try:
            _, info = env.reset()
            np.testing.assert_array_equal(info["task_id"], np.arange(104))
            for step in range(16):
                actions = np.tile([step / 16, 0, step % 2], (104, 1)).astype(
                    np.float32
                )
                _, _, _, _, info = env.step(actions)
                np.testing.assert_array_equal(info["task_id"], np.arange(104))
        finally:
            env.close()

    def test_slot_horizons_and_threshold_validation(self) -> None:
        """Continuous slots retain independent horizons and finite thresholds."""
        env = make_gymnasium(
            "Pong-v5",
            continuous=True,
            num_envs=3,
            env_task_names=["pong", "breakout", "space_invaders"],
            env_max_episode_steps=[2, 3, 5],
            seed=[11, 12, 13],
            noop_max=1,
            use_fire_reset=False,
            frame_skip=1,
        )
        try:
            env.reset()
            for step in range(1, 31):
                _, _, terminal, truncated, info = env.step(
                    np.zeros((3, 3), np.float32)
                )
                expected = np.array([step % (limit + 1) for limit in (2, 3, 5)])
                np.testing.assert_array_equal(info["elapsed_step"], expected)
                np.testing.assert_array_equal(truncated, expected == [2, 3, 5])
                self.assertFalse(np.any(terminal))
        finally:
            env.close()
        for threshold in (-2, 0, 0.25, 2):
            spec = make_spec(
                "Pong-v5",
                continuous=True,
                continuous_action_threshold=threshold,
            )
            self.assertEqual(spec.config.continuous_action_threshold, threshold)
        for threshold in (np.nan, np.inf, -np.inf):
            with self.assertRaisesRegex(ValueError, "threshold must be finite"):
                make_spec(
                    "Pong-v5",
                    continuous=True,
                    continuous_action_threshold=threshold,
                )

    def test_dm_life_loss_and_discount(self) -> None:
        """Life-loss discounts and soft resets agree with a standalone slot."""
        kwargs: dict[str, Any] = dict(
            continuous=True,
            episodic_life=True,
            zero_discount_on_life_loss=True,
            reward_clip=True,
            noop_max=3,
        )
        mixed = make_multitask(
            ["Breakout-v5", "Skiing-v5"], env_type="dm", seed=[31, 41], **kwargs
        )
        single = make_dm("Breakout-v5", seed=31, num_threads=1, **kwargs)
        try:
            previous_lives = mixed.reset().observation.lives[0]
            single.reset()
            losses = 0
            for _ in range(1600):
                actions = np.array([[0, 0, 0], [1, 0, 0]], np.float32)
                actual, expected = mixed.step(actions), single.step(actions[:1])
                for field in ("step_type", "reward", "discount"):
                    np.testing.assert_array_equal(
                        getattr(actual, field)[0], getattr(expected, field)[0]
                    )
                for field in ("obs", "ram", "lives", "task_id"):
                    np.testing.assert_array_equal(
                        getattr(actual.observation, field)[0],
                        getattr(expected.observation, field)[0],
                    )
                lives = actual.observation.lives[0]
                if 0 < lives < previous_lives:
                    losses += 1
                    self.assertEqual(actual.discount[0], 0)
                previous_lives = lives
            self.assertGreater(losses, 0)
        finally:
            mixed.close()
            single.close()


if __name__ == "__main__":
    absltest.main()
