HumanoidBench
=============

This fork runs HumanoidBench task logic in C++ on EnvPool's worker threads.
It provides 112 robot/task IDs with Gymnasium, ``dm_env``, state and pixel
interfaces. Flat control is supported.

Install a published wheel
-------------------------

The published binary release is the ``1.2.5+apex.hb2`` GitHub prerelease from
source ``e1323c3``:
`download the hb2 wheels <https://github.com/wzhhasadream/envpool/releases/tag/wheels-1.2.5-apex.hb2-e1323c3>`_.
It contains 16 wheels for standard CPython 3.11, 3.12, 3.13 and 3.14 across
Linux x86_64, Linux ``aarch64``, macOS arm64 and Windows AMD64. Choose the wheel
matching your Python version and platform; these are not free-threaded Python
builds. Each wheel includes MuJoCo 3.6.0, model assets and native plugins.
The original Python HumanoidBench package and a compiler are unnecessary
when using a published wheel.

This release retains HumanoidBench and adds comprehensive compatible
:doc:`DMC mixed-task pools <multitask>`. It is not published to PyPI; the
existing ``hb1`` wheels remain unchanged and lack the expanded DMC support.
All 16 ``hb2`` wheel build/install/release-test jobs passed. The separate
`comprehensive Bazel run <https://github.com/wzhhasadream/envpool/actions/runs/39372019108>`_
passed 116/116 targets on each Linux architecture and 109/116 on Windows;
macOS was cancelled at 112/116 with four failed targets already recorded.
DMC mixed-task cases passed 162/162 on Windows and 159/162 on macOS, where
three exact pixel comparisons failed. See
`known limitations <https://github.com/wzhhasadream/envpool/blob/main/KNOWN_LIMITATIONS.md>`_
for remaining GL/render/GLFW failures and the Windows ViZDoom timeout.
The wheel matrix does not establish full cross-platform oracle equivalence.

Basic usage
-----------

.. code-block:: python

   import envpool
   import numpy as np

   env = envpool.make_gymnasium(
       "h1-walk-v0", num_envs=16, num_threads=8, seed=0,
       friction_multiplier=0.5,
   )
   obs, info = env.reset()
   actions = np.zeros((16, 19), np.float32)
   obs, reward, terminated, truncated, info = env.step(actions)
   env.close()

Use ``from_pixels=True, render_width=84, render_height=84`` for pixels,
and ``envpool.make_dm`` for the ``dm_env`` interface. The native
``friction_multiplier`` scales all three friction components of every
geometry at creation; its default is 1.0.

Historical HumanoidBench validation
-----------------------------------

The following records the earlier ``hb1``-era Linux x86_64 / CPython 3.11 oracle
validation. These results are retained as historical evidence, not a fresh
``hb2`` HumanoidBench test run or a claim of full-episode oracle equivalence
across every published wheel platform.

The oracle is HumanoidBench 0.2.1 at
``d198f7b49cf889d86e41fdcba6b174e6b235fd12`` from
`humanoid-bench-compatible <https://github.com/wzhhasadream/humanoid-bench-compatible>`_.
The comparison used the same official MuJoCo binary as the native
implementation. Initial physics and task state were synchronized once after
reset. Subsequent steps received identical float32 actions without further
synchronization.

* 112 native IDs reset and stepped successfully.
* 111 oracle comparisons had zero state observation error for up to 128 steps
  or the first termination. The original ``g1-truck-v0`` reward raises a
  ``KeyError`` for the missing ``free_base`` joint; the native task uses the
  root position.
* All 45 ApexRL task IDs passed state comparisons and four post-step pixel
  comparisons at 84 by 84 pixels, with zero pixel error.
* 27 targeted task-state transitions passed reward, observation, metric and
  termination checks.
* 51 DMC tasks passed Gymnasium and ``dm_env`` reset/step checks with
  NumPy 1.26.0. These were single-task checks, not the ``hb2`` mixed-task suite.

EnvPool rewards are float32. Historical diagnostic per-timestep reward
comparisons differed by at most 1.78e-15 before the public float32 conversion.
Native reset RNG is deterministic across thread counts but does not reproduce
NumPy's random stream, so identical seeds do not imply identical oracle
initial states.
Pixel coverage in that validation was four post-step frames per selected
task. That run did not validate full-episode alignment or other operating
systems. JAX JIT was tested on CPU with JAX and ``jaxlib`` 0.6.0. The existence
of the later ``hb1`` and ``hb2`` wheel releases does not broaden these historical oracle
claims. See :doc:`dm_control` and :doc:`multitask` for the current DMC scope.

Build from source
-----------------

The developer build script below is a Linux x86_64 / CPython 3.11 path, not
the full release-wheel matrix. Building the current checkout produces the
current source version; it does not download a published ``hb2`` wheel.
It requires a C++ toolchain, ``curl``, ``patchelf``, ``setuptools``, ``wheel``
and the EnvPool runtime dependencies. Install ``jaxlib`` 0.6.0 to reproduce the tested FFI build.
From the repository root, run:

.. code-block:: bash

   PYTHON=python bash scripts/build_humanoid_bench.sh

The script downloads the pinned assets and official MuJoCo binary, rebuilds
HumanoidBench and DMC, preserves the other native families from EnvPool 1.2.5,
and installs the resulting wheel. Source builds download the Python oracle
package only to obtain its pinned assets. Runtime task logic remains native.

Licenses
--------

The task translation retains the HumanoidBench MIT license. EnvPool integration
uses Apache-2.0. Model assets retain their individual licenses, and the wheel
includes MuJoCo's license notices. See the packaged ``LICENSE`` and ``NOTICE``
files under ``envpool/mujoco/humanoid_bench``.
