"""Generate native task metadata from the pinned HumanoidBench installation."""

import ast
import json
import inspect
import textwrap
from pathlib import Path

import humanoid_bench.env as upstream
import mujoco

root = Path(upstream.__file__).parent
destination = Path(__file__).resolve().parents[2] / "envpool/mujoco/humanoid_bench"
assets = destination / "assets"
task_metrics = {}
for task_name, task_cls in upstream.TASKS.items():
    fields = set()
    for name, method in inspect.getmembers(task_cls, inspect.isfunction):
        if not name.startswith(("get_reward", "get_terminated")):
            continue
        for node in ast.walk(ast.parse(textwrap.dedent(inspect.getsource(method)))):
            if isinstance(node, ast.Dict):
                fields.update(key.value for key in node.keys if isinstance(key, ast.Constant) and isinstance(key.value, str))
            if isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name) and node.value.id == "reward_dict" and isinstance(node.slice, ast.Constant):
                fields.add(node.slice.value)
    task_metrics[task_name] = sorted(fields)
metrics = sorted(set().union(*task_metrics.values()))

tasks = {}
for robot_name, robot_cls in upstream.ROBOTS.items():
    robot = robot_cls()
    for task_name, task_cls in upstream.TASKS.items():
        control = "torque" if robot_name == "g1" else "pos"
        xml = f"envs/{robot_name}_{control}_{task_name}.xml"
        if not (assets / xml).exists():
            continue
        try:
            model = mujoco.MjModel.from_xml_path(str(assets / xml))
        except ValueError as error:
            print(f"Invalid upstream model {robot_name}-{task_name}: {error}")
            continue
        task = task_cls(robot, None)
        tasks[f"{robot_name}-{task_name}-v0"] = dict(
            task=task_name, xml_file=xml, robot_dof=robot.dof,
            obs_dim=task.observation_space.shape[0], action_dim=model.nu,
            nq=model.nq, nv=model.nv, nbody=model.nbody, ngeom=model.ngeom,
            frame_skip=task.frame_skip, max_episode_steps=task.max_episode_steps,
            camera_name=task.camera_name, stand_height=1.28 if robot_name == "g1" else 1.65,
            reset_noise_scale=0.0 if task_name in {"cube", "bookshelf_simple", "bookshelf_hard", "kitchen"} else 0.01,
            reward_threshold=task.success_bar,
        )

destination.mkdir(parents=True, exist_ok=True)
(destination / "registry.json").write_text(json.dumps(dict(
    source="https://github.com/wzhhasadream/humanoid-bench-compatible",
    revision="d198f7b49cf889d86e41fdcba6b174e6b235fd12",
    version="0.2.1", metrics=metrics, task_metrics=task_metrics, tasks=tasks,
), indent=2) + "\n")
print(f"Generated {len(tasks)} tasks and {len(metrics)} metric fields")
