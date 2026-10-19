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
"""Optional adapters against native Gymnasium pools and explicit subset reset."""

from __future__ import annotations

from contextlib import ExitStack
from copy import deepcopy
from typing import Any

import gymnasium as gym
import numpy as np
from absl.testing import absltest, parameterized

import envpool.atari.registration  # noqa: F401
import envpool.mujoco.dmc.registration  # noqa: F401
import envpool.mujoco.gym.registration  # noqa: F401
import envpool.mujoco.metaworld.registration  # noqa: F401
import envpool.mujoco.myosuite.registration  # noqa: F401
from envpool.python.glfw_context import preload_windows_gl_dlls
from envpool.registration import make_gymnasium, make_multitask
from envpool.wrappers import PixelObservationLayout, SameStepAutoReset

preload_windows_gl_dlls(strict=True)


def _equal(actual: Any, expected: Any) -> None:
    if isinstance(expected, dict):
        for key, value in expected.items():
            _equal(actual[key], value)
    else:
        np.testing.assert_array_equal(actual, expected)


def _row(value: Any, index: int) -> Any:
    if isinstance(value, dict):
        return {key: _row(child, index) for key, child in value.items()}
    return value[index]


def _pixels(value: Any, frames: int, channels: int) -> Any:
    if isinstance(value, dict):
        return {
            key: _pixels(child, frames, channels)
            for key, child in value.items()
        }
    if value.dtype != np.uint8 or value.ndim < 3:
        return value
    return np.moveaxis(
        value.reshape(*value.shape[:-3], frames, channels, *value.shape[-2:]),
        -3,
        -1,
    )


class NativeWrappersTest(parameterized.TestCase):
    """Bitwise explicit-reset parity, task IDs, success flags, and render."""

    @parameterized.named_parameters(
        ("gym", ("HalfCheetah-v4",), {}),
        ("dmc", ("DogRun-v1", "DogStand-v1"), {}),
        (
            "myosuite",
            ("myoFingerReachFixed-v0", "myoFatiFingerReachRandom-v0"),
            {},
        ),
        ("metaworld", ("MetaWorld/Reach-v3", "MetaWorld/Push-v3"), {}),
        ("atari_gray", ("Pong-v5", "Breakout-v5"), {"gray_scale": True}),
        ("atari_rgb", ("Pong-v5", "Breakout-v5"), {"gray_scale": False}),
    )
    def test_pixels_both_orders_and_multitask(
        self, tasks: tuple[str, ...], extra: dict[str, Any]
    ) -> None:
        """Compare every transition with a native pool reset explicitly by ID."""
        atari = "gray_scale" in extra
        channels = 1 if extra.get("gray_scale") else 3
        options: dict[str, Any] = dict(
            seed=[31, 47],
            num_threads=1,
            render_mode="rgb_array",
            max_episode_steps=5,
        )
        if atari:
            options.update(
                stack_num=2,
                frame_skip=1,
                img_width=24,
                img_height=20,
                full_action_space=True,
                noop_max=1,
                use_fire_reset=False,
                **extra,
            )
        else:
            options.update(
                from_pixels=True,
                frame_stack=2,
                action_repeat=2,
                render_width=24,
                render_height=20,
            )
        for pixel_outer in (False, True):
            with self.subTest(pixel_outer=pixel_outer), ExitStack() as stack:

                def create() -> Any:
                    if len(tasks) == 1:
                        return make_gymnasium(tasks[0], num_envs=2, **options)
                    return make_multitask(tasks, **options)

                native, reference = create(), create()
                wrapped = (
                    PixelObservationLayout(SameStepAutoReset(native))
                    if pixel_outer
                    else SameStepAutoReset(PixelObservationLayout(native))
                )
                stack.callback(wrapped.close)
                stack.callback(reference.close)
                obs, info = wrapped.reset()
                expected, expected_info = reference.reset()
                _equal(obs, _pixels(expected, 2, channels))
                _equal(info, expected_info)
                self.assertEqual(
                    wrapped.single_observation_space.shape,
                    (2, 20, 24, channels),
                )
                if len(tasks) > 1:
                    self.assertEqual(wrapped.env_task_ids, tasks)
                truncations = 0
                for step in range(16):
                    ids = np.array([1] if step == 4 else [1, 0], np.int32)
                    space = native.single_action_space
                    action = (
                        np.zeros(len(ids), np.int32)
                        if isinstance(space, gym.spaces.Discrete)
                        else np.zeros((len(ids), *space.shape), space.dtype)
                    )
                    expected, rew, term, trunc, expected_info = deepcopy(
                        reference.step(action, ids)
                    )
                    obs, actual_rew, actual_term, actual_trunc, info = (
                        wrapped.step(action, ids)
                    )
                    _equal(actual_rew, rew)
                    _equal(actual_term, term)
                    _equal(actual_trunc, trunc)
                    _equal(info["env_id"], ids)
                    done = term | trunc
                    truncations += int(trunc.sum())
                    if np.any(done):
                        _equal(info["_final_obs"], done)
                        _equal(info["_final_info"], done)
                        reset_obs, reset_info = reference.reset(ids[done])
                        reset_positions = {
                            eid: row
                            for row, eid in enumerate(reset_info["env_id"])
                        }
                    for row, eid in enumerate(ids):
                        if done[row]:
                            _equal(
                                info["final_obs"][row],
                                _pixels(_row(expected, row), 2, channels),
                            )
                            _equal(
                                _row(info["final_info"], row),
                                _row(expected_info, row),
                            )
                            reset_row = reset_positions[eid]
                            _equal(
                                _row(obs, row),
                                _pixels(
                                    _row(reset_obs, reset_row), 2, channels
                                ),
                            )
                            _equal(_row(info, row), _row(reset_info, reset_row))
                            self.assertEqual(info["elapsed_step"][row], 0)
                        else:
                            _equal(
                                _row(obs, row),
                                _pixels(_row(expected, row), 2, channels),
                            )
                            _equal(_row(info, row), _row(expected_info, row))
                        if "success" in expected_info:
                            reported = info["final_info"] if done[row] else info
                            self.assertEqual(
                                reported["success"][row],
                                expected_info["success"][row],
                            )
                    if step == 8:
                        options_reset = {"reset_mask": np.array([True, False])}
                        reset_obs, reset_info = wrapped.reset(
                            options=options_reset
                        )
                        ref_obs, ref_info = reference.reset(
                            options=options_reset
                        )
                        _equal(reset_obs, _pixels(ref_obs, 2, channels))
                        _equal(reset_info, ref_info)
                self.assertGreater(truncations, 2)
                _equal(
                    wrapped.render(env_ids=[1, 0]),
                    reference.render(env_ids=[1, 0]),
                )

    def test_native_termination_is_not_lost(self) -> None:
        """A real terminal InvertedPendulum transition returns reset obs and reward."""
        wrapped = SameStepAutoReset(
            make_gymnasium("InvertedPendulum-v4", num_envs=1, seed=5)
        )
        try:
            wrapped.reset()
            for _ in range(200):
                obs, reward, term, trunc, info = wrapped.step(
                    np.full((1, 1), 3.0, np.float64)
                )
                if term[0] or trunc[0]:
                    break
            self.assertTrue(term[0])
            self.assertFalse(trunc[0])
            self.assertEqual(reward[0], 1.0)
            self.assertEqual(info["elapsed_step"][0], 0)
            self.assertGreater(info["final_info"]["elapsed_step"][0], 0)
            self.assertFalse(np.array_equal(obs[0], info["final_obs"][0]))
            self.assertEqual(
                wrapped.step(np.full((1, 1), 3.0, np.float64))[4][
                    "elapsed_step"
                ][0],
                1,
            )
        finally:
            wrapped.close()

    def test_native_terminal_success_belongs_to_final_info(self) -> None:
        """Apex can read a real positive success after same-step reset clears it."""
        native = make_gymnasium(
            "myoTorsoPoseFixed-v0",
            num_envs=1,
            seed=5,
            action_repeat=3,
            max_episode_steps=1,
        )
        wrapped = SameStepAutoReset(native)
        try:
            wrapped.reset()
            action = np.zeros_like(wrapped.action_space.sample())
            for _ in range(2):
                _, _, term, trunc, info = wrapped.step(action)
                self.assertFalse(term[0])
                self.assertTrue(trunc[0])
                self.assertEqual(info["success"][0], 0.0)
                final = info["final_info"]
                self.assertEqual(final["success"][0], 1.0)
                self.assertEqual(final["success"][0], final["solved"][0])
                self.assertTrue(final["_success"][0])
                self.assertEqual(final["task_id"][0], info["task_id"][0])
        finally:
            wrapped.close()

    def test_native_async_pixel_routing_and_same_step_rejection(self) -> None:
        """Pixel layout preserves asynchronous IDs; autoreset rejects the pool."""
        pool = make_gymnasium(
            "Pong-v5",
            num_envs=4,
            batch_size=2,
            num_threads=1,
            stack_num=2,
            img_height=20,
            img_width=24,
            noop_max=1,
        )
        try:
            with self.assertRaisesRegex(ValueError, "synchronous"):
                SameStepAutoReset(pool)
            wrapped = PixelObservationLayout(pool)
            wrapped.async_reset()
            for _ in range(8):
                obs, _, _, _, info = wrapped.recv()
                self.assertEqual(obs.shape, (2, 2, 20, 24, 1))
                ids = info["env_id"]
                self.assertLen(np.unique(ids), 2)
                wrapped.send(np.zeros(2, np.int32), ids)
        finally:
            pool.close()


if __name__ == "__main__":
    absltest.main()
