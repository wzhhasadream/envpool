# SPDX-License-Identifier: Apache-2.0

"""Task IDs generated from the pinned upstream robot/task registry."""

import json
from pathlib import Path

from envpool.registration import register

root = Path(__file__).parent
registry = json.loads((root / "registry.json").read_text())
for task_id, config in registry["tasks"].items():
    register(
        task_id=task_id,
        import_path="envpool.mujoco.humanoid_bench",
        spec_cls="HumanoidBenchEnvSpec",
        dm_cls="HumanoidBenchDMEnvPool",
        gymnasium_cls="HumanoidBenchGymnasiumEnvPool",
        base_path=str(root),
        metric_names=registry["metrics"],
        **config,
    )
