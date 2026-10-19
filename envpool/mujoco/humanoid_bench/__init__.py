"""Native HumanoidBench environments, using EnvPool's C++ worker pool."""

import json
import ctypes
from pathlib import Path

from envpool.mujoco.humanoid_bench_envpool import (
    _HumanoidBenchEnvPool, _HumanoidBenchEnvSpec,
    _HumanoidBenchPixelEnvPool, _HumanoidBenchPixelEnvSpec,
)
from envpool.python.api import py_env

# The native module loads MuJoCo before registering its bundled plugins.
_plugins = [ctypes.CDLL(str(path)) for path in sorted((Path(__file__).parent / "plugins").glob("*.so"))]

HumanoidBenchEnvSpec, HumanoidBenchDMEnvPool, HumanoidBenchGymnasiumEnvPool = py_env(_HumanoidBenchEnvSpec, _HumanoidBenchEnvPool)
HumanoidBenchPixelEnvSpec, HumanoidBenchPixelDMEnvPool, HumanoidBenchPixelGymnasiumEnvPool = py_env(_HumanoidBenchPixelEnvSpec, _HumanoidBenchPixelEnvPool)

_registry = json.loads(Path(__file__).with_name("registry.json").read_text())
_metric_indices = {name: index for index, name in enumerate(_registry["metrics"])}


def _with_task_info(pool):
    convert = pool._to

    def to_gymnasium(self, state_values, reset, return_info):
        result = convert(self, state_values, reset, return_info)
        info = result[-1]
        metrics = info.pop("metrics")
        for name in _registry["task_metrics"][self.spec.config.task]:
            info[name] = metrics[:, _metric_indices[name]]
        terminated = info.pop("terminated")
        if reset:
            return result
        return result[0], result[1], terminated, result[3], info

    pool._to = to_gymnasium


_with_task_info(HumanoidBenchGymnasiumEnvPool)
_with_task_info(HumanoidBenchPixelGymnasiumEnvPool)
