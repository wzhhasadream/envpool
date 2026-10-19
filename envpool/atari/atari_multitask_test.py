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
"""Bitwise native mixed-ROM pool parity against standalone ALE 0.11.2 slots."""

from typing import Any

import numpy as np
from absl.testing import absltest, parameterized

import envpool.atari.registration as reg
from envpool.registration import (
    make_dm,
    make_gymnasium,
    make_multitask,
    make_spec,
)

_TASKS = ("Pong-v5", "Breakout-v5", "SpaceInvaders-v5")
# ALE excludes FIRE variants from both games' legal-action sets.
_NONCANONICAL_GAMES = {"lost_luggage", "skiing"}


def _public_id(rom: str) -> str:
    return "".join(part.capitalize() for part in rom.split("_")) + "-v5"


class AtariMultitaskTest(parameterized.TestCase):
    """Each mixed slot must be exactly its independently seeded single game."""

    def assert_state_equal(self, mixed: tuple, row: int, single: tuple) -> None:
        """Compare a received slot, excluding pool-local routing IDs."""
        self.assertLen(mixed, len(single))
        for value, expected in zip(mixed[:-1], single[:-1], strict=True):
            np.testing.assert_array_equal(value[row], expected[0])
        for key in (
            "task_id",
            "lives",
            "reward",
            "terminated",
            "ram",
            "elapsed_step",
        ):
            np.testing.assert_array_equal(
                mixed[-1][key][row], single[-1][key][0]
            )

    @parameterized.named_parameters(
        ("defaults", {}),
        (
            "rgb_single_frame",
            dict(
                frame_skip=1,
                stack_num=1,
                gray_scale=False,
                img_height=210,
                img_width=160,
                noop_max=1,
                use_fire_reset=False,
            ),
        ),
        (
            "sticky_clipped_lives",
            dict(
                frame_skip=5,
                stack_num=2,
                repeat_action_probability=0.25,
                reward_clip=True,
                episodic_life=True,
                zero_discount_on_life_loss=True,
                noop_max=7,
            ),
        ),
        (
            "linear_resize",
            dict(
                frame_skip=2,
                stack_num=3,
                gray_scale=False,
                img_height=48,
                img_width=64,
                use_inter_area_resize=False,
                mode=0,
                difficulty=0,
            ),
        ),
    )
    def test_rollout_subset_reset_and_render(
        self, options: dict[str, Any]
    ) -> None:
        """Shared preprocessing and subset resets preserve standalone rollouts."""
        seeds = [71, 903, 22, 118]
        slot_tasks = (_TASKS[0], _TASKS[1], _TASKS[1], _TASKS[2])
        kwargs: dict[str, Any] = dict(
            full_action_space=True, render_mode="rgb_array", **options
        )
        mixed = make_multitask(
            _TASKS,
            envs_per_task=[1, 2, 1],
            seed=seeds,
            num_threads=3,
            max_episode_steps=79,
            **kwargs,
        )
        singles = [
            make_gymnasium(
                task, seed=seed, num_threads=1, max_episode_steps=79, **kwargs
            )
            for task, seed in zip(slot_tasks, seeds, strict=True)
        ]
        try:
            self.assertEqual(mixed.single_action_space.n, 18)
            self.assertEqual(mixed.env_task_ids, slot_tasks)
            actual = mixed.reset()
            for row, single in enumerate(singles):
                self.assert_state_equal(actual, row, single.reset())
            rng = np.random.default_rng(914)
            for step in range(256):
                actions = rng.integers(0, 18, size=4, dtype=np.int32)
                actual = mixed.step(actions)
                for row, single in enumerate(singles):
                    self.assert_state_equal(
                        actual, row, single.step(actions[row : row + 1])
                    )
                if step % 37 == 0:
                    ids = np.array([3, 1], dtype=np.int32)
                    actual = mixed.reset(ids)
                    np.testing.assert_array_equal(actual[-1]["env_id"], ids)
                    for row, slot in enumerate(ids):
                        self.assert_state_equal(
                            actual, row, singles[slot].reset()
                        )
                if step % 32 == 0:
                    ids = [3, 0, 2]
                    frames = mixed.render(env_ids=ids)
                    self.assertIsNotNone(frames)
                    for row, slot in enumerate(ids):
                        reference = singles[slot].render()
                        assert frames is not None and reference is not None
                        np.testing.assert_array_equal(frames[row], reference[0])
        finally:
            mixed.close()
            for single in singles:
                single.close()

    def test_every_compatible_game_has_canonical_id_and_matches_standalone(
        self,
    ) -> None:
        """Every canonical-18 ROM keeps its identity and standalone trajectory."""
        # Cover all compatible games, including games that have no lives. Small
        # groups bound thread/memory use while still mixing different ROMs.
        compatible_games = [
            game
            for game in reg.atari_game_list
            if game not in _NONCANONICAL_GAMES
        ]
        for start in range(0, len(compatible_games), 4):
            games = compatible_games[start : start + 4]
            tasks = tuple(map(_public_id, games))
            seeds = [201 + start + i for i in range(len(tasks))]
            kwargs: dict[str, Any] = dict(
                full_action_space=True,
                frame_skip=4,
                stack_num=1,
                img_height=42,
                img_width=42,
                noop_max=3,
                render_mode="rgb_array",
                max_episode_steps=83,
            )
            mixed = make_multitask(tasks, seed=seeds, num_threads=2, **kwargs)
            singles = [
                make_gymnasium(task, seed=seed, num_threads=1, **kwargs)
                for task, seed in zip(tasks, seeds, strict=True)
            ]
            try:
                actual = mixed.reset()
                np.testing.assert_array_equal(
                    actual[-1]["task_id"],
                    [reg.atari_game_list.index(game) for game in games],
                )
                for row, single in enumerate(singles):
                    self.assert_state_equal(actual, row, single.reset())
                for step in range(128):
                    actions = (
                        np.arange(len(tasks), dtype=np.int32) * 5 + step * 7
                    ) % 18
                    actual = mixed.step(actions)
                    for row, single in enumerate(singles):
                        with self.subTest(task=tasks[row], step=step):
                            self.assert_state_equal(
                                actual, row, single.step(actions[row : row + 1])
                            )
                    if step in (0, 63, 127):
                        frames = mixed.render(env_ids=list(range(len(tasks))))
                        for row, single in enumerate(singles):
                            reference = single.render()
                            assert frames is not None and reference is not None
                            np.testing.assert_array_equal(
                                frames[row], reference[0]
                            )
            finally:
                mixed.close()
                for single in singles:
                    single.close()

    def test_async_batches_preserve_slot_mapping(self) -> None:
        """Completion order never changes game, seed, or recurrent state."""
        counts = [2, 3, 1]
        tasks = tuple(
            task
            for task, count in zip(_TASKS, counts, strict=True)
            for _ in range(count)
        )
        seeds = [63, 8, 92, 401, 78, 36]
        kwargs: dict[str, Any] = dict(
            full_action_space=True,
            repeat_action_probability=0.25,
            max_episode_steps=19,
        )
        mixed = make_multitask(
            _TASKS,
            envs_per_task=counts,
            seed=seeds,
            batch_size=2,
            num_threads=3,
            **kwargs,
        )
        singles = [
            make_gymnasium(task, seed=seed, num_threads=1, **kwargs)
            for task, seed in zip(tasks, seeds, strict=True)
        ]
        try:
            pending = {}
            for slot, single in enumerate(singles):
                obs, info = single.reset()
                pending[slot] = (
                    obs,
                    np.zeros(1, np.float32),
                    np.zeros(1, bool),
                    np.zeros(1, bool),
                    info,
                )
            mixed.async_reset()
            received = np.zeros(6, dtype=int)
            for _ in range(300):
                actual = mixed.recv()
                ids = actual[-1]["env_id"]
                actions = np.empty(2, np.int32)
                for row, slot in enumerate(ids):
                    self.assert_state_equal(actual, row, pending[slot])
                    received[slot] += 1
                    actions[row] = (received[slot] * 7 + slot * 3) % 18
                    pending[slot] = singles[slot].step(actions[row : row + 1])
                mixed.send(actions, env_id=ids)
            self.assertTrue(np.all(received > 0), received)
        finally:
            mixed.close()
            for single in singles:
                single.close()

    @parameterized.parameters(False, True)
    def test_dm_life_loss_discount_matches_single_game(
        self, episodic_life: bool
    ) -> None:
        """Life loss, soft resets, game-over, and discount stay slot-local."""
        kwargs: dict[str, Any] = dict(
            full_action_space=True,
            episodic_life=episodic_life,
            zero_discount_on_life_loss=True,
            reward_clip=True,
            noop_max=3,
            frame_skip=4,
            max_episode_steps=5000,
        )
        mixed = make_multitask(
            _TASKS[:2], env_type="dm", seed=[41, 31], **kwargs
        )
        singles = [
            make_dm(task, seed=seed, num_threads=1, **kwargs)
            for task, seed in zip(_TASKS[:2], [41, 31], strict=True)
        ]
        try:
            current = mixed.reset()
            for single in singles:
                single.reset()
            previous_lives = current.observation.lives.copy()
            life_losses = 0
            for _ in range(1600):
                current = mixed.step(np.zeros(2, np.int32))
                for slot, single in enumerate(singles):
                    expected = single.step(np.zeros(1, np.int32))
                    for field in ("step_type", "reward", "discount"):
                        np.testing.assert_array_equal(
                            getattr(current, field)[slot],
                            getattr(expected, field)[0],
                        )
                    for field in (
                        "obs",
                        "lives",
                        "reward",
                        "terminated",
                        "ram",
                        "task_id",
                    ):
                        np.testing.assert_array_equal(
                            getattr(current.observation, field)[slot],
                            getattr(expected.observation, field)[0],
                        )
                lives = current.observation.lives
                if 0 < lives[1] < previous_lives[1]:
                    life_losses += 1
                    self.assertEqual(current.discount[1], 0)
                    self.assertEqual(bool(current.last()[1]), episodic_life)
                previous_lives = lives.copy()
            self.assertGreater(life_losses, 0)
        finally:
            mixed.close()
            for single in singles:
                single.close()

    def test_native_slot_horizons_and_legacy_defaults(self) -> None:
        """Native slot limits are independent and legacy minimal spaces remain."""
        env = make_gymnasium(
            "Pong-v5",
            num_envs=3,
            seed=[11, 12, 13],
            env_task_names=["pong", "breakout", "space_invaders"],
            env_max_episode_steps=[2, 3, 5],
            full_action_space=True,
            noop_max=1,
            use_fire_reset=False,
            frame_skip=1,
        )
        try:
            env.reset()
            for step in range(1, 31):
                _, _, terminated, truncated, info = env.step(
                    np.zeros(3, np.int32)
                )
                expected = np.array([
                    step % (horizon + 1) for horizon in (2, 3, 5)
                ])
                np.testing.assert_array_equal(info["elapsed_step"], expected)
                np.testing.assert_array_equal(truncated, expected == [2, 3, 5])
                self.assertFalse(np.any(terminated))
        finally:
            env.close()
        legacy = make_spec("Breakout-v5")
        self.assertFalse(legacy.config.full_action_space)
        self.assertEqual(legacy.action_space.n, 4)
        self.assertEmpty(legacy.config.env_task_names)
        self.assertEmpty(legacy.config.env_max_episode_steps)

    @parameterized.parameters(
        ({"env_task_names": ["pong", "breakout"]}, "full_action_space=True"),
        ({"env_task_names": ["pong"], "full_action_space": True}, "one task"),
        (
            {"env_task_names": ["pong", "missing"], "full_action_space": True},
            "Unknown Atari task",
        ),
        ({"env_max_episode_steps": [3]}, "one horizon"),
        ({"env_max_episode_steps": [3, 0]}, "positive"),
    )
    def test_native_validation(
        self, kwargs: dict[str, Any], message: str
    ) -> None:
        """Malformed native mappings fail before worker construction."""
        with self.assertRaisesRegex(ValueError, message):
            make_spec("Pong-v5", num_envs=2, **kwargs)

    @parameterized.parameters(sorted(_NONCANONICAL_GAMES))
    def test_noncanonical_games_are_rejected_without_changing_legacy(
        self, game: str
    ) -> None:
        """Games that forbid FIRE retain their existing nine-action space."""
        task = _public_id(game)
        with self.assertRaisesRegex(ValueError, "canonical 18 ALE actions"):
            make_multitask([task])
        legacy = make_gymnasium(task, seed=51, full_action_space=True)
        explicit_empty = make_gymnasium(
            task, seed=51, full_action_space=True, env_task_names=[]
        )
        try:
            self.assertEqual(legacy.single_action_space.n, 9)
            actual = legacy.reset()
            self.assert_state_equal(actual, 0, explicit_empty.reset())
            self.assertEqual(
                actual[-1]["task_id"][0], reg.atari_game_list.index(game)
            )
            for step in range(128):
                action = np.array([(step * 5) % 9], dtype=np.int32)
                self.assert_state_equal(
                    legacy.step(action), 0, explicit_empty.step(action)
                )
        finally:
            legacy.close()
            explicit_empty.close()

    def test_public_full_action_requirement_and_task_order(self) -> None:
        """Public defaults are safe and game indices ignore request ordering."""
        with self.assertRaisesRegex(ValueError, "full_action_space"):
            make_multitask(_TASKS, full_action_space=False)
        first = make_multitask(_TASKS)
        reverse = make_multitask(_TASKS[::-1])
        try:
            self.assertEqual(first.single_action_space.n, 18)
            np.testing.assert_array_equal(
                first.reset()[-1]["task_id"],
                reverse.reset()[-1]["task_id"][::-1],
            )
        finally:
            first.close()
            reverse.close()


if __name__ == "__main__":
    absltest.main()
