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
"""Tests for the dm_control render path."""

import ctypes
import os
import platform
import subprocess
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any, cast

import numpy as np
from absl.testing import absltest

from envpool.python.glfw_context import preload_windows_gl_dlls

preload_windows_gl_dlls(strict=True)


def _configure_linux_mujoco_gl() -> None:
    if platform.system() != "Linux":
        return
    if os.environ.get("MUJOCO_GL"):
        if os.environ["MUJOCO_GL"] == "egl":
            os.environ.setdefault("EGL_PLATFORM", "surfaceless")
        return
    for backend in ("egl", "osmesa"):
        env = dict(os.environ)
        env["MUJOCO_GL"] = backend
        if backend == "egl":
            env.setdefault("EGL_PLATFORM", "surfaceless")
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                (
                    "import mujoco; "
                    "ctx = mujoco.GLContext(1, 1); "
                    "ctx.make_current(); "
                    "ctx.free()"
                ),
            ],
            env=env,
            check=False,
            capture_output=True,
        )
        if result.returncode == 0:
            os.environ["MUJOCO_GL"] = backend
            if backend == "egl":
                os.environ.setdefault("EGL_PLATFORM", "surfaceless")
            return


_configure_linux_mujoco_gl()

from dm_control import _render, suite
from dm_control._render import base as dm_control_render_base
from dm_control._render import executor as dm_control_render_executor
from dm_control.mujoco import engine as dm_control_engine
from dm_control.mujoco.wrapper.mjbindings import mjlib

import envpool.mujoco.dmc.registration as reg
from envpool.registration import make_dm, make_gym

_RENDER_STEPS = 3


def _configure_macos_dm_control_renderer() -> None:
    if platform.system() != "Darwin":
        return

    class _CglContext(dm_control_render_base.ContextBase):
        def __init__(self, max_width: int, max_height: int):
            super().__init__(
                max_width,
                max_height,
                dm_control_render_executor.PassthroughRenderExecutor,
            )

        def _platform_init(self, max_width: int, max_height: int) -> None:
            del max_width, max_height
            from mujoco.cgl import cgl

            attrib = cgl.CGLPixelFormatAttribute
            profile = cgl.CGLOpenGLProfile
            attrib_values = (
                attrib.CGLPFAOpenGLProfile,
                profile.CGLOGLPVersion_Legacy,
                attrib.CGLPFAColorSize,
                24,
                attrib.CGLPFAAlphaSize,
                8,
                attrib.CGLPFADepthSize,
                24,
                attrib.CGLPFAStencilSize,
                8,
                attrib.CGLPFAAllowOfflineRenderers,
                0,
                0,  # terminator
            )
            attribs = (ctypes.c_int * len(attrib_values))(*attrib_values)
            self._pixel_format = cgl.CGLPixelFormatObj()
            num_pixel_formats = cgl.GLint()
            cgl.CGLChoosePixelFormat(
                attribs,
                ctypes.byref(self._pixel_format),
                ctypes.byref(num_pixel_formats),
            )
            if not self._pixel_format or num_pixel_formats.value == 0:
                raise RuntimeError("failed to create CGL pixel format")

            self._context = cgl.CGLContextObj()
            cgl.CGLCreateContext(
                self._pixel_format,
                0,
                ctypes.byref(self._context),
            )
            if not self._context:
                cgl.CGLReleasePixelFormat(self._pixel_format)
                self._pixel_format = None
                raise RuntimeError("failed to create CGL context")
            self._locked = False

        def _platform_make_current(self) -> None:
            from mujoco.cgl import cgl

            cgl.CGLSetCurrentContext(self._context)
            # Mirror mujoco.cgl.GLContext so the official renderer uses the
            # same CGL lifecycle as EnvPool's native renderer.
            if not self._locked:
                cgl.CGLLockContext(self._context)
                self._locked = True

        def _platform_free(self) -> None:
            from mujoco.cgl import cgl

            if self._context:
                if self._locked:
                    cgl.CGLUnlockContext(self._context)
                    self._locked = False
                cgl.CGLSetCurrentContext(None)
                cgl.CGLReleaseContext(self._context)
                self._context = None
            if self._pixel_format:
                cgl.CGLReleasePixelFormat(self._pixel_format)
                self._pixel_format = None

    _render.Renderer = _CglContext
    _render.BACKEND = "cgl"
    _render.USING_GPU = True


_configure_macos_dm_control_renderer()


def _task_map() -> dict[str, tuple[str, str]]:
    result: dict[str, tuple[str, str]] = {}
    for domain, task, _ in reg.dmc_mujoco_envs:
        domain_name = "".join(g[:1].upper() + g[1:] for g in domain.split("_"))
        task_name = "".join(g[:1].upper() + g[1:] for g in task.split("_"))
        result[f"{domain_name}{task_name}-v1"] = (domain, task)
    return result


def _render_array(env: Any, env_ids: Any = None) -> np.ndarray:
    frame = env.render(env_ids=env_ids)
    assert frame is not None
    return cast(np.ndarray, frame)


def _zero_action(space: Any, num_envs: int) -> np.ndarray:
    sample = np.asarray(space.sample())
    zero = np.zeros_like(sample)
    if sample.ndim == 0:
        return np.full((num_envs,), zero.item(), dtype=sample.dtype)
    return np.repeat(zero[np.newaxis, ...], num_envs, axis=0)


def _assert_frames_close(
    actual: np.ndarray,
    expected: np.ndarray,
    *,
    max_mean_abs_diff: float = 1.0,
    max_mismatch_ratio: float = 0.1,
) -> None:
    if actual.shape != expected.shape:
        raise AssertionError(
            f"frame shapes differ: {actual.shape} != {expected.shape}"
        )
    diff = np.abs(actual.astype(np.int16) - expected.astype(np.int16))
    if diff.size == 0:
        return
    mismatch_ratio = float(np.count_nonzero(diff)) / float(diff.size)
    mean_abs_diff = float(diff.mean())
    if mean_abs_diff > max_mean_abs_diff:
        raise AssertionError(
            "mean render delta "
            f"{mean_abs_diff:.3f} exceeded {max_mean_abs_diff:.3f}"
        )
    if mismatch_ratio > max_mismatch_ratio:
        raise AssertionError(
            "render mismatch ratio "
            f"{mismatch_ratio:.4%} exceeded {max_mismatch_ratio:.4%}"
        )


@contextmanager
def _without_dm_control_render_contexts() -> Iterator[None]:
    """Prevent non-rendering upstream resets from initializing OpenGL."""
    original_contexts = dm_control_engine.Physics.contexts
    dm_control_engine.Physics.contexts = property(lambda _: None)
    try:
        yield
    finally:
        dm_control_engine.Physics.contexts = original_contexts


def _load_official_env(domain: str, task: str) -> Any:
    task_kwargs = {"time_limit": 30} if domain == "lqr" else {}
    if domain == "quadruped" and task == "escape":
        with _without_dm_control_render_contexts():
            return suite.load(domain, task, task_kwargs=task_kwargs)
    return suite.load(domain, task, task_kwargs=task_kwargs)


def _reset_official_state(env: Any, ts: Any, domain: str, task: str) -> None:
    # Initialize task-side effects (for example Finger/Fish visibility) and
    # clear dm_control's reset-next-step flag before syncing the initial state.
    # physics.reset_context alone only resets mjData, not the task/environment.
    if domain == "quadruped" and task == "escape":
        with _without_dm_control_render_contexts():
            env.reset()
    else:
        env.reset()
    with env.physics.reset_context():
        env.physics.data.qpos = ts.observation.qpos0[0]
        if hasattr(ts.observation, "qvel0"):
            env.physics.data.qvel = ts.observation.qvel0[0]
        if hasattr(ts.observation, "act0"):
            env.physics.data.act = ts.observation.act0[0]
        if domain == "cheetah":
            for _ in range(200):
                env.physics.step()
            env.physics.data.time = 0
        elif domain == "cartpole":
            env.physics.data.qvel = ts.observation.qvel0[0]
        elif domain == "reacher":
            target = ts.observation.target[0]
            env.physics.named.model.geom_pos["target", "x"] = target[0]
            env.physics.named.model.geom_pos["target", "y"] = target[1]
        elif domain == "swimmer":
            xpos, ypos = ts.observation.target0[0]
            env.physics.named.model.geom_pos["target", "x"] = xpos
            env.physics.named.model.geom_pos["target", "y"] = ypos
            env.physics.named.model.light_pos["target_light", "x"] = xpos
            env.physics.named.model.light_pos["target_light", "y"] = ypos
        elif domain == "fish" and task == "swim":
            target = ts.observation.target0[0]
            env.physics.named.model.geom_pos["target", "x"] = target[0]
            env.physics.named.model.geom_pos["target", "y"] = target[1]
            env.physics.named.model.geom_pos["target", "z"] = target[2]
        elif domain in {"finger", "ball_in_cup", "humanoid"}:
            if domain == "finger" and task in {"turn_easy", "turn_hard"}:
                target_angle = ts.observation.target[0][0]
                hinge = env.physics.named.data.xanchor["hinge", ["x", "z"]]
                radius = env.physics.named.model.geom_size["cap1"].sum()
                target_x = hinge[0] + radius * np.sin(target_angle)
                target_z = hinge[1] + radius * np.cos(target_angle)
                env.physics.named.model.site_pos["target", ["x", "z"]] = (
                    target_x,
                    target_z,
                )
            env.physics.after_reset()
        elif domain == "point_mass":
            env.physics.model.wrap_prm = ts.observation.wrap_prm[0]
        elif domain == "manipulator":
            use_peg = "peg" in task
            insert = "insert" in task
            obj = "peg" if use_peg else "ball"
            target = "target_" + obj
            obj_joints = [obj + "_x", obj + "_z", obj + "_y"]
            receptacle = "slot" if use_peg else "cup"
            (
                target_x,
                target_z,
                target_angle,
                init_type,
                object_x,
                object_z,
                object_angle,
                qvel_objx,
            ) = ts.observation.random_info[0]
            p = env.physics.named
            if insert:
                p.model.body_pos[receptacle, ["x", "z"]] = (
                    target_x,
                    target_z,
                )
                p.model.body_quat[receptacle, ["qw", "qy"]] = (
                    np.cos(target_angle / 2),
                    np.sin(target_angle / 2),
                )
            p.model.body_pos[target, ["x", "z"]] = target_x, target_z
            p.model.body_quat[target, ["qw", "qy"]] = (
                np.cos(target_angle / 2),
                np.sin(target_angle / 2),
            )
            if np.isclose(init_type, 2):
                env.physics.after_reset()
            else:
                p.data.qvel[obj + "_x"] = qvel_objx
            p.data.qpos[obj_joints] = object_x, object_z, object_angle
            env.physics.after_reset()
        if domain == "lqr":
            env.physics.model.jnt_stiffness[:] = ts.observation.stiffness0[0]
        if domain == "quadruped" and task == "escape":
            hfield = ts.observation.hfield0[0]
            start_idx = env.physics.model.hfield_adr[0]
            env.physics.model.hfield_data[
                start_idx : start_idx + len(hfield)
            ] = hfield
        if domain == "stacker":
            target_x, target_z = ts.observation.target0[0]
            env.physics.named.model.body_pos["target", ["x", "z"]] = (
                target_x,
                target_z,
            )
        if domain == "humanoid_CMU":
            env.physics.after_reset()
    if hasattr(ts.observation, "qacc_warmstart0"):
        env.physics.data.qacc_warmstart = ts.observation.qacc_warmstart0[0]
    if domain == "quadruped" and task == "escape":
        # A previous render may already have uploaded the old episode's terrain.
        # Updating model data alone does not refresh that OpenGL heightfield.
        contexts = env.physics._contexts
        if contexts is not None:
            with contexts.gl.make_current() as ctx:
                ctx.call(
                    mjlib.mjr_uploadHField,
                    env.physics.model.ptr,
                    contexts.mujoco.ptr,
                    0,
                )


class MujocoDmcRenderTest(absltest.TestCase):
    """Render regression tests for dm_control-backed MuJoCo tasks."""

    def test_rgb_array_render_is_batch_consistent(self) -> None:
        """Batch render output should match per-env renders for WalkerWalk."""
        env = make_gym(
            "WalkerWalk-v1",
            num_envs=2,
            seed=0,
            render_mode="rgb_array",
            render_width=96,
            render_height=72,
        )
        try:
            env.reset()
            for step_idx in range(_RENDER_STEPS):
                frame0 = _render_array(env)
                frame1 = _render_array(env, env_ids=1)
                frames = _render_array(env, env_ids=[0, 1])
                frame0_again = _render_array(env)
                self.assertEqual(frame0.shape, (1, 72, 96, 3))
                self.assertEqual(frame1.shape, (1, 72, 96, 3))
                self.assertEqual(frames.shape, (2, 72, 96, 3))
                self.assertEqual(frames.dtype, np.uint8)
                _assert_frames_close(frame0[0], frames[0])
                _assert_frames_close(frame1[0], frames[1])
                _assert_frames_close(frame0, frame0_again)
                if step_idx + 1 < _RENDER_STEPS:
                    env.step(_zero_action(env.single_action_space, 2))
        finally:
            env.close()

    def test_render_succeeds_for_all_tasks(self) -> None:
        """Every dm_control-backed task should render through the render path."""
        for task_id in sorted(_task_map()):
            with self.subTest(task_id=task_id):
                env = make_gym(
                    task_id,
                    num_envs=1,
                    seed=0,
                    render_mode="rgb_array",
                    render_width=96,
                    render_height=72,
                )
                try:
                    env.reset()
                    for step_idx in range(_RENDER_STEPS):
                        frame = _render_array(env)[0]
                        frame_again = _render_array(env)[0]
                        self.assertEqual(frame.shape, (72, 96, 3))
                        self.assertEqual(frame_again.shape, (72, 96, 3))
                        self.assertEqual(frame.dtype, np.uint8)
                        self.assertEqual(frame_again.dtype, np.uint8)
                        self.assertGreater(
                            int(frame.max()) - int(frame.min()), 0
                        )
                        self.assertGreater(
                            int(frame_again.max()) - int(frame_again.min()), 0
                        )
                        if step_idx + 1 < _RENDER_STEPS:
                            env.step(_zero_action(env.single_action_space, 1))
                finally:
                    env.close()

    def test_render_matches_official_reset_frame_for_all_tasks(self) -> None:
        """Rendered reset frames should match dm_control after state sync."""
        for task_id, (domain, task) in sorted(_task_map().items()):
            with self.subTest(task_id=task_id):
                env = make_dm(
                    task_id,
                    num_envs=1,
                    seed=0,
                    render_mode="rgb_array",
                    render_width=96,
                    render_height=72,
                )
                oracle = _load_official_env(domain, task)
                try:
                    ts = env.reset(np.array([0]))
                    _reset_official_state(oracle, ts, domain, task)
                    frame = _render_array(env)[0]
                    expected = cast(
                        np.ndarray,
                        oracle.physics.render(
                            height=72,
                            width=96,
                            camera_id=-1,
                        ),
                    )
                    _assert_frames_close(
                        frame,
                        expected,
                        max_mean_abs_diff=8.0,
                        max_mismatch_ratio=0.12,
                    )
                finally:
                    env.close()
                    oracle.close()

    def test_repeated_pixels_match_official_trajectory(self) -> None:
        """Compare every task's repeated pixels after a single initial sync."""
        for task_id, (domain, task) in sorted(_task_map().items()):
            camera_id = 2 if domain == "quadruped" else 0
            with self.subTest(task_id=task_id):
                env = make_dm(
                    task_id,
                    num_envs=1,
                    seed=0,
                    from_pixels=True,
                    action_repeat=3,
                    frame_stack=3,
                    render_mode="rgb_array",
                    render_width=96,
                    render_height=72,
                )
                oracle = _load_official_env(domain, task)
                try:
                    # Escape changes its heightfield on every reset; exercise
                    # reuse of both native and original rendering contexts.
                    resets = (
                        2 if (domain, task) == ("quadruped", "escape") else 1
                    )
                    for _ in range(resets):
                        ts = env.reset(np.array([0]))
                        _reset_official_state(oracle, ts, domain, task)
                        # The camera is intentionally omitted from make_dm:
                        # assert the ApexRL domain default, not an explicit override.
                        self.assertEqual(
                            env.config["render_camera_id"], camera_id
                        )
                        frame = oracle.physics.render(
                            height=72, width=96, camera_id=camera_id
                        )
                        frames = [frame] * 3
                        rng = np.random.default_rng(17)
                        for step in range(_RENDER_STEPS + 1):
                            if step:
                                action = rng.uniform(
                                    -0.3, 0.3, oracle.action_spec().shape
                                ).astype(oracle.action_spec().dtype)
                                for _ in range(3):
                                    oracle.step(action)
                                ts = env.step(action[np.newaxis, ...])
                                frame = oracle.physics.render(
                                    height=72, width=96, camera_id=camera_id
                                )
                                frames = [*frames[1:], frame]
                            self.assertTrue(
                                ts.first()[0] if step == 0 else ts.mid()[0]
                            )
                            self.assertAlmostEqual(
                                oracle.physics.data.time,
                                3 * step * oracle.control_timestep(),
                            )
                            pixels = ts.observation.pixels[0]
                            self.assertEqual(pixels.shape, (9, 72, 96))
                            self.assertEqual(pixels.dtype, np.uint8)
                            expected = np.concatenate(
                                [f.transpose(2, 0, 1) for f in frames], axis=0
                            )
                            # Use the existing renderer-level budget, without the
                            # broader reset-only oracle tolerance above. No state
                            # synchronization is performed during this trajectory.
                            _assert_frames_close(pixels, expected)
                            np.testing.assert_array_equal(
                                pixels[-3:].transpose(1, 2, 0),
                                _render_array(env)[0],
                            )
                finally:
                    env.close()
                    oracle.close()


if __name__ == "__main__":
    absltest.main()
