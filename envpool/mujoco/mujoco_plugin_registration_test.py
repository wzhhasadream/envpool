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
"""Regression tests for MuJoCo process-global plugin registration."""

import subprocess
import sys

from absl.testing import absltest


def _assert_imports_succeed(
    *modules: str, task_ids: tuple[str, ...] = ()
) -> None:
    # Disable eager family imports so each subprocess really exercises the
    # requested native-module order rather than envpool.entry's fixed order.
    code = """
import sys
import types
sys.modules["envpool.entry"] = types.ModuleType("envpool.entry")
"""
    code += "\n".join(f"import {module}" for module in modules)
    if task_ids:
        # The minimal Bazel target omits envpool's package-level API exports.
        code += f"""
from envpool.registration import make_gymnasium
import envpool.mujoco.humanoid_bench.registration
for task_id in {task_ids!r}:
    env = make_gymnasium(task_id, num_envs=1, num_threads=1)
    try:
        env.reset()
        env.step(env.action_space.sample() * 0)
    finally:
        env.close()
"""
    subprocess.run([sys.executable, "-c", code], check=True)


class MujocoPluginRegistrationTest(absltest.TestCase):
    """Tests that MuJoCo decoder plugins are safe across extension modules."""

    def test_humanoid_bench_registers_plugins_standalone(self) -> None:
        """Loads STL, OBJ, and touch-grid models without another native module."""
        _assert_imports_succeed(
            "envpool.mujoco.humanoid_bench",
            task_ids=("h1-walk-v0", "h1hand-walk-v0", "h1touch-walk-v0"),
        )

    def test_humanoid_bench_then_existing_modules_share_plugins(self) -> None:
        """Checks other MuJoCo imports preserve all HumanoidBench plugins."""
        _assert_imports_succeed(
            "envpool.mujoco.humanoid_bench",
            "envpool.mujoco.robotics",
            "envpool.mujoco.dmc",
            "envpool.mujoco.playground",
            task_ids=("h1-walk-v0", "h1hand-walk-v0", "h1touch-walk-v0"),
        )

    def test_existing_modules_then_humanoid_bench_share_plugins(self) -> None:
        """Checks HumanoidBench also registers its plugins when imported last."""
        _assert_imports_succeed(
            "envpool.mujoco.robotics",
            "envpool.mujoco.dmc",
            "envpool.mujoco.playground",
            "envpool.mujoco.humanoid_bench",
            task_ids=("h1-walk-v0", "h1hand-walk-v0", "h1touch-walk-v0"),
        )

    def test_robotics_then_dmc_imports_share_stl_decoder(self) -> None:
        """Checks the release-test import order that first exposed the issue."""
        _assert_imports_succeed("envpool.mujoco.robotics", "envpool.mujoco.dmc")

    def test_dmc_then_robotics_imports_share_stl_decoder(self) -> None:
        """Checks the opposite import order for standalone DMC users."""
        _assert_imports_succeed("envpool.mujoco.dmc", "envpool.mujoco.robotics")

    def test_playground_shares_stl_decoder_with_existing_modules(self) -> None:
        """Checks Playground can coexist with the existing STL users."""
        _assert_imports_succeed(
            "envpool.mujoco.playground",
            "envpool.mujoco.dmc",
            "envpool.mujoco.robotics",
        )

    def test_robotics_then_metaworld_imports_share_obj_decoder(self) -> None:
        """Checks the release-test import order with MetaWorld linked in."""
        _assert_imports_succeed(
            "envpool.mujoco.robotics", "envpool.mujoco.metaworld"
        )

    def test_metaworld_then_robotics_imports_share_obj_decoder(self) -> None:
        """Checks the opposite import order for standalone MetaWorld users."""
        _assert_imports_succeed(
            "envpool.mujoco.metaworld", "envpool.mujoco.robotics"
        )


if __name__ == "__main__":
    absltest.main()
