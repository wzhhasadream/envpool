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
"""Typed Atari XLA actions and synchronous FFI validation errors."""

import platform
from typing import Any

import jax
import numpy as np
from absl.testing import absltest, parameterized

import envpool.atari.registration  # noqa: F401
from envpool.registration import make_multitask


class AtariContinuousXlaTest(parameterized.TestCase):
    """XLA retains the ordinary worker queue and reports invalid batch errors."""

    def _unavailable(self, env: Any) -> bool:
        system = platform.system()
        if system not in ("Darwin", "Windows"):
            return False
        name = "macOS" if system == "Darwin" else "Windows"
        with self.assertRaisesRegex(RuntimeError, f"unavailable on {name}"):
            env.xla()
        return True

    @parameterized.parameters(False, True)
    def test_cpu_xla_matches_python(self, continuous: bool) -> None:
        """Both typed action spaces reproduce the ordinary Python send path."""
        kwargs: dict[str, Any] = dict(
            continuous=continuous,
            seed=[101, 203],
            num_threads=2,
            noop_max=1,
            max_episode_steps=17,
            repeat_action_probability=0.25,
        )
        actual = make_multitask(["Pong-v5", "Breakout-v5"], **kwargs)
        reference = make_multitask(["Pong-v5", "Breakout-v5"], **kwargs)
        try:
            if self._unavailable(actual):
                return
            actual.reset()
            reference.reset()
            handle, _, _, step = actual.xla()
            compiled_step = jax.jit(step, backend="cpu")
            for i in range(100):
                if continuous:
                    actions = np.array(
                        [[(i % 11) / 10, 0.3, i % 2], [0.7, -0.5, 1]],
                        np.float32,
                    )
                else:
                    actions = np.array([i % 18, (i * 3) % 18], np.int32)
                handle, result = compiled_step(handle, actions)
                result = jax.device_get(result)
                expected = reference.step(actions)
                for value, wanted in zip(
                    result[:-1], expected[:-1], strict=True
                ):
                    np.testing.assert_array_equal(value, wanted)
                for key in ("ram", "task_id", "elapsed_step", "lives"):
                    np.testing.assert_array_equal(
                        result[-1][key], expected[-1][key]
                    )
        finally:
            actual.close()
            reference.close()

    @parameterized.parameters(np.nan, np.inf, -0.1, 1.1)
    def test_invalid_xla_batch_is_atomic(self, radius: float) -> None:
        """A rejected FFI send cannot advance the valid first slot."""
        kwargs: dict[str, Any] = dict(
            continuous=True, seed=[101, 203], num_threads=1, noop_max=1
        )
        actual = make_multitask(["Breakout-v5", "Skiing-v5"], **kwargs)
        reference = make_multitask(["Breakout-v5", "Skiing-v5"], **kwargs)
        try:
            if self._unavailable(actual):
                return
            actual.reset()
            reference.reset()
            handle, _, send, _ = actual.xla()
            compiled_send = jax.jit(send, backend="cpu")
            actions = np.array([[1, 0, 1], [radius, 0, 0]], np.float32)
            with self.assertRaisesRegex(Exception, "Atari continuous actions"):
                jax.block_until_ready(compiled_send(handle, actions))
            valid = np.ones((2, 3), np.float32)
            result, expected = actual.step(valid), reference.step(valid)
            for value, wanted in zip(result[:-1], expected[:-1], strict=True):
                np.testing.assert_array_equal(value, wanted)
            for key in ("ram", "task_id", "elapsed_step", "lives"):
                np.testing.assert_array_equal(
                    result[-1][key], expected[-1][key]
                )
        finally:
            actual.close()
            reference.close()


if __name__ == "__main__":
    absltest.main()
