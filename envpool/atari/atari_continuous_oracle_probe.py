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
"""Isolated official ALE 0.11.2 / NumPy 2.4.3 continuous rollout oracle."""

import argparse

import ale_py
import numpy as np
from ale_py.env import AtariEnv


class _ActionSink:
    def __init__(self) -> None:
        self.action = 0
        self.strength = 0.0

    def act(self, action: int, strength: float) -> float:
        self.action = int(action)
        self.strength = float(strength)
        return 0.0

    def game_over(self, with_truncation: bool = False) -> bool:
        return False

    def game_truncated(self) -> bool:
        return False


def _conversion_probe(
    actions: np.ndarray, threshold: float
) -> tuple[np.ndarray, np.ndarray]:
    # Execute the original AtariEnv.step conversion with only the emulator
    # replaced by a recorder. This deliberately keeps NumPy's actual trig.
    env = AtariEnv.__new__(AtariEnv)
    env._frameskip = 1
    env.continuous = True
    env.continuous_action_threshold = threshold
    sink = _ActionSink()
    env.ale = sink
    env._get_obs = lambda: None
    env._get_info = dict
    indices, strengths = [], []
    for action in actions:
        env.step(action)
        indices.append(sink.action)
        strengths.append(sink.strength)
    return np.asarray(indices, np.int32), np.asarray(strengths, np.float32)


def main() -> None:
    """Run the original Gymnasium ALE converter and emulator in Python 3.12."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--convert-only", action="store_true")
    args = parser.parse_args()
    if ale_py.__version__ != "0.11.2" or np.__version__ != "2.4.3":
        raise RuntimeError("Expected ALE 0.11.2 and NumPy 2.4.3 oracle")
    data = np.load(args.input)
    if args.convert_only:
        indices, strengths = _conversion_probe(
            data["actions"], float(data["threshold"])
        )
        np.savez(args.output, indices=indices, strengths=strengths)
        return
    games, seeds, actions = data["games"], data["seeds"], data["actions"]
    count, slots = actions.shape[:2]
    obs = np.empty((count + 1, slots, 3, 210, 160), np.uint8)
    ram = np.empty((count + 1, slots, 128), np.uint8)
    reward = np.zeros((count + 1, slots), np.float32)
    lives = np.zeros((count + 1, slots), np.int32)
    terminated = np.zeros((count + 1, slots), bool)
    truncated = np.zeros((count + 1, slots), bool)
    game_over = np.zeros((count + 1, slots), bool)
    elapsed = np.zeros((count + 1, slots), np.int32)
    for slot, (game, seed) in enumerate(zip(games, seeds, strict=True)):
        env = AtariEnv(
            game=str(game),
            obs_type="rgb",
            frameskip=1,
            continuous=True,
            full_action_space=False,
            render_mode="rgb_array",
            repeat_action_probability=float(data["sticky"]),
            continuous_action_threshold=float(data["threshold"]),
        )
        try:
            # One reset-time seed/bootstrap synchronization. EnvPool assigns
            # the given ALE seed directly and performs one no-op on reset.
            # All subsequent emulator changes use the original step/reset API.
            env.ale.setInt("random_seed", int(seed))
            env.load_game()
            done = True
            steps = 0
            for step in range(count + 1):
                if done:
                    env.reset()
                    env.ale.act(ale_py.Action.NOOP)
                    if env.ale.game_over():
                        env.reset()
                    steps = 0
                    done = False
                else:
                    _, value, terminal, trunc, _ = env.step(
                        actions[step - 1, slot]
                    )
                    steps += 1
                    trunc = trunc or steps >= int(data["horizon"])
                    reward[step, slot] = value
                    terminated[step, slot] = terminal and not trunc
                    truncated[step, slot] = trunc
                    done = terminal or trunc
                screen = env.ale.getScreenRGB()
                np.testing.assert_array_equal(env.render(), screen)
                # Existing native Atari preprocessing/rendering retains the
                # first 210 rows even for PAL ROMs with 250-row ALE screens.
                # Keep that legacy crop explicit; do not rescale PAL pixels.
                if screen.shape[0] < 210 or screen.shape[1:] != (160, 3):
                    raise ValueError(
                        f"Unexpected ALE RGB shape: {screen.shape}"
                    )
                obs[step, slot] = screen[:210].transpose(2, 0, 1)
                ram[step, slot] = env.ale.getRAM()
                lives[step, slot] = env.ale.lives()
                game_over[step, slot] = env.ale.game_over()
                elapsed[step, slot] = steps
        finally:
            env.close()
    np.savez_compressed(
        args.output,
        obs=obs,
        ram=ram,
        reward=reward,
        lives=lives,
        terminated=terminated,
        truncated=truncated,
        game_over=game_over,
        elapsed=elapsed,
    )


if __name__ == "__main__":
    main()
