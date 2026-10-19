HumanoidBench
=============

The ``1.2.5+apex.hb1`` fork runs HumanoidBench task logic in C++ on
EnvPool's worker threads. It provides 112 robot/task IDs with Gymnasium,
``dm_env``, state and pixel interfaces. Flat control is supported.

Install the CPython 3.11 Linux x86_64 wheel from the
`GitHub release <https://github.com/wzhhasadream/envpool/releases/tag/v1.2.5-apex-hb1>`_.
The wheel includes MuJoCo 3.6.0, model assets and native plugins.
The original Python HumanoidBench package and a compiler are unnecessary
when using the wheel.

.. code-block:: python

   import envpool
   import numpy as np

   env = envpool.make_gymnasium("h1-walk-v0", num_envs=16, num_threads=8, seed=0, friction_multiplier=0.5)
   obs, info = env.reset()
   obs, reward, terminated, truncated, info = env.step(np.zeros((16, 19), np.float32))
   env.close()

Use ``from_pixels=True, render_width=84, render_height=84`` for pixels,
and ``envpool.make_dm`` for the ``dm_env`` interface. The native
``friction_multiplier`` scales all three friction components of every
geometry at creation; its default is 1.0.

Validation
----------

The oracle is HumanoidBench 0.2.1 at
``d198f7b49cf889d86e41fdcba6b174e6b235fd12`` from
`humanoid-bench-compatible <https://github.com/wzhhasadream/humanoid-bench-compatible>`_.
It uses the same official MuJoCo binary as the native implementation.
Initial physics and task state are synchronized once after reset. Subsequent
steps receive identical float32 actions without further synchronization.

* 112 native IDs reset and step successfully.
* 111 oracle comparisons have zero state observation error for up to 128 steps
  or the first termination. The original ``g1-truck-v0`` reward raises a
  ``KeyError`` for the missing ``free_base`` joint; the native task uses the
  root position.
* All 45 ApexRL task IDs pass state comparisons and four post-step pixel
  comparisons at 84 by 84 pixels, with zero pixel error.
* 27 targeted task-state transitions pass reward, observation, metric and
  termination checks.
* 51 DMC tasks pass Gymnasium and ``dm_env`` reset/step checks with NumPy 1.26.0.

EnvPool rewards are float32. Diagnostic per-timestep reward comparisons differ
by at most 1.78e-15 before the public float32 conversion. Native reset RNG is
deterministic across thread counts but does not reproduce NumPy's random
stream, so identical seeds do not imply identical oracle initial states.
Pixel coverage is four post-step frames per selected task. Full-episode
alignment and other operating systems have not been validated. JAX JIT was
tested on CPU with JAX/JAXlib 0.6.0.

Build from source
-----------------

The developer build requires Linux x86_64, CPython 3.11, a C++ toolchain,
``curl``, ``patchelf``, ``setuptools``, ``wheel`` and the EnvPool runtime
dependencies. Install JAXlib 0.6.0 to reproduce the tested FFI build.
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
