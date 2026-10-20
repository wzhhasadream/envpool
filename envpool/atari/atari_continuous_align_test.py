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
"""All-game mixed continuous rollouts against the pinned original ALE wrapper."""

import os
import subprocess
import sys
import tempfile
from pathlib import Path
from unittest import mock

import numpy as np
from absl import logging
from absl.testing import absltest, parameterized
from envpool.atari.atari_envpool import _continuous_action

import envpool.atari.registration as reg
from envpool.registration import make_multitask


def _oracle_command() -> list[str]:
    override = os.environ.get("ENVPOOL_ATARI_ORACLE_PROBE")
    if override:
        return [override]
    executable = Path(__file__).with_name("atari_continuous_oracle_probe")
    # Bazel's extensionless launcher is a Python script, not a Windows binary.
    # Use the native launcher to preserve the oracle's isolated Python runtime.
    if sys.platform == "win32":
        executable = executable.with_suffix(".exe")
    if executable.is_file():
        return [str(executable)]
    # Direct local runs may use an interpreter containing the same pinned
    # oracle packages. Bazel always uses the isolated Python 3.12 executable.
    return [
        sys.executable,
        str(Path(__file__).with_name("atari_continuous_oracle_probe.py")),
    ]


class OracleCommandTest(parameterized.TestCase):
    """Select executable launchers without changing the oracle runtime."""

    @parameterized.parameters("win32", "linux", "darwin")
    def test_explicit_override_takes_precedence(self, platform: str) -> None:
        """Keep an explicit command intact, including paths containing spaces."""
        override = "custom oracle path/oracle.exe"
        with (
            mock.patch.dict(os.environ, ENVPOOL_ATARI_ORACLE_PROBE=override),
            mock.patch.object(sys, "platform", platform),
            mock.patch.object(Path, "is_file") as is_file,
        ):
            self.assertEqual(_oracle_command(), [override])
        is_file.assert_not_called()

    @parameterized.named_parameters(
        ("windows_both_launchers", "win32", ("", ".exe"), ".exe"),
        ("windows_exe_only", "win32", (".exe",), ".exe"),
        ("linux_both_launchers", "linux", ("", ".exe"), ""),
        ("macos_both_launchers", "darwin", ("", ".exe"), ""),
        ("linux_extensionless_only", "linux", ("",), ""),
        ("windows_script_only", "win32", ("",), ".py"),
        ("windows_no_launcher", "win32", (), ".py"),
        ("linux_exe_only", "linux", (".exe",), ".py"),
        ("linux_no_launcher", "linux", (), ".py"),
        ("macos_no_launcher", "darwin", (), ".py"),
    )
    def test_platform_launcher_selection(
        self,
        platform: str,
        available_suffixes: tuple[str, ...],
        expected_suffix: str,
    ) -> None:
        """Never pass a script directly to Windows subprocess creation."""
        executable = Path(__file__).with_name("atari_continuous_oracle_probe")
        available = {
            executable.with_suffix(suffix) for suffix in available_suffixes
        }
        expected = [str(executable.with_suffix(expected_suffix))]
        if expected_suffix == ".py":
            expected.insert(0, sys.executable)
        with (
            mock.patch.dict(os.environ, ENVPOOL_ATARI_ORACLE_PROBE=""),
            mock.patch.object(sys, "platform", platform),
            mock.patch.object(
                Path,
                "is_file",
                autospec=True,
                side_effect=available.__contains__,
            ),
        ):
            self.assertEqual(_oracle_command(), expected)


class AtariContinuousAlignTest(absltest.TestCase):
    """No state synchronization after the initial reset/bootstrap alignment."""

    def test_all_registered_games_in_mixed_pools(self) -> None:
        """All 104 games match the original continuous wrapper, including FIRE."""
        for start in range(0, len(reg.atari_game_list), 4):
            games = reg.atari_game_list[start : start + 4]
            with self.subTest(games=games):
                self._compare_rollout(
                    games, steps=256, horizon=83, seed=1100 + start
                )

    def test_long_paddle_and_illegal_fire_rollout(self) -> None:
        """Paddles and nine-legal-action games match for a nontrivial trajectory."""
        for games in (["breakout", "kaboom"], ["lost_luggage", "skiing"]):
            with self.subTest(games=games):
                self._compare_rollout(games, steps=1024, horizon=300, seed=1701)

    def test_custom_thresholds(self) -> None:
        """Finite thresholds outside the Box are not silently clamped."""
        for threshold in (-0.25, 0.0, 0.25, 1.25):
            with self.subTest(threshold=threshold):
                self._compare_rollout(
                    ["breakout", "skiing"],
                    steps=64,
                    horizon=23,
                    seed=1801,
                    threshold=threshold,
                )

    def test_characterize_numpy_threshold_rounding(self) -> None:
        """Report the known NumPy/libm boundary distinction without tolerance."""
        actions = np.array(
            [
                [0.6937808, 0.766001, 0],
                [0.84240776, 0.93534994, 0],
                [0.5, 0, 0.5],
            ],
            dtype=np.float32,
        )
        with tempfile.TemporaryDirectory() as directory:
            source, output = (
                Path(directory) / "actions.npz",
                Path(directory) / "result.npz",
            )
            np.savez(source, actions=actions, threshold=np.float32(0.5))
            subprocess.run(
                [
                    *_oracle_command(),
                    "--input",
                    str(source),
                    "--output",
                    str(output),
                    "--convert-only",
                ],
                check=True,
                timeout=60,
            )
            with np.load(output) as original:
                native = np.array([
                    _continuous_action(action, 0.5)[0] for action in actions
                ])
                # This is a characterization, not a parity/tolerance assertion.
                # A NumPy build using scalar libm may agree on every row.
                logging.warning(
                    "Continuous threshold precision report: actions=%s native=%s original_numpy=%s",
                    actions.tolist(),
                    native.tolist(),
                    original["indices"].tolist(),
                )
                np.testing.assert_array_equal(
                    original["strengths"], actions[:, 0]
                )
                self.assertEqual(native[-1], original["indices"][-1])
                artifact_dir = os.environ.get("TEST_UNDECLARED_OUTPUTS_DIR")
                if artifact_dir:
                    np.savez(
                        Path(artifact_dir)
                        / "continuous_threshold_rounding.npz",
                        actions=actions,
                        native=native,
                        original=original["indices"],
                    )

    def _compare_rollout(
        self,
        games: list[str],
        steps: int,
        horizon: int,
        seed: int,
        threshold: float = 0.5,
    ) -> None:
        task_ids = [
            "".join(part.capitalize() for part in game.split("_")) + "-v5"
            for game in games
        ]
        seeds = np.arange(seed, seed + len(games), dtype=np.int32)
        rng = np.random.default_rng(seed)
        actions = rng.uniform(
            [0, -np.pi, 0], [1, np.pi, 1], (steps, len(games), 3)
        ).astype(np.float32)
        # Include strict fire equality, zero radius, cardinal boundaries, and
        # FIRE variants that ALE converts to NOOP for LostLuggage and Skiing.
        representatives = np.asarray(
            [
                [0, 0, 0],
                [0.5, 0, 0.5],
                [0.5, 0, 1],
                [1, 0, 0],
                [1, np.pi / 2, 1],
                [1, -np.pi, 1],
            ],
            np.float32,
        )
        for step, action in enumerate(representatives):
            actions[step] = action
        with tempfile.TemporaryDirectory() as directory:
            input_path = Path(directory) / "input.npz"
            output_path = Path(directory) / "oracle.npz"
            np.savez(
                input_path,
                games=np.asarray(games),
                seeds=seeds,
                actions=actions,
                sticky=np.float32(0.25),
                threshold=np.float32(threshold),
                horizon=horizon,
            )
            subprocess.run(
                [
                    *_oracle_command(),
                    "--input",
                    str(input_path),
                    "--output",
                    str(output_path),
                ],
                check=True,
                timeout=300,
            )
            # Materialize each compressed array once, not once per step.
            with np.load(output_path) as archive:
                expected = {key: archive[key] for key in archive.files}
            env = make_multitask(
                task_ids,
                continuous=True,
                continuous_action_threshold=threshold,
                full_action_space=False,
                seed=seeds.tolist(),
                num_threads=2,
                frame_skip=1,
                stack_num=1,
                gray_scale=False,
                img_height=210,
                img_width=160,
                noop_max=1,
                use_fire_reset=False,
                repeat_action_probability=0.25,
                max_episode_steps=horizon,
                render_mode="rgb_array",
            )
            try:
                obs, info = env.reset()
                for step in range(steps + 1):
                    if step:
                        obs, reward, terminal, trunc, info = env.step(
                            actions[step - 1]
                        )
                        np.testing.assert_array_equal(
                            reward, expected["reward"][step]
                        )
                        np.testing.assert_array_equal(
                            terminal, expected["terminated"][step]
                        )
                        np.testing.assert_array_equal(
                            trunc, expected["truncated"][step]
                        )
                    np.testing.assert_array_equal(obs, expected["obs"][step])
                    for key, oracle_key in (
                        ("ram", "ram"),
                        ("lives", "lives"),
                        ("terminated", "game_over"),
                        ("elapsed_step", "elapsed"),
                    ):
                        np.testing.assert_array_equal(
                            info[key], expected[oracle_key][step]
                        )
                    if step in (0, steps // 2, steps):
                        np.testing.assert_array_equal(
                            env.render(env_ids=list(range(len(games)))),
                            expected["obs"][step].transpose(0, 2, 3, 1),
                        )
            finally:
                env.close()


if __name__ == "__main__":
    absltest.main()
