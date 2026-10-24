<div align="center">
  <a href="http://envpool.readthedocs.io"><img width="666px" height="auto" src="https://envpool.readthedocs.io/en/latest/_static/envpool-logo.png"></a>
</div>

---

[![PyPI](https://img.shields.io/pypi/v/envpool)](https://pypi.org/project/envpool/) [![Downloads](https://static.pepy.tech/personalized-badge/envpool?period=total&units=international_system&left_color=grey&right_color=orange&left_text=PyPI%20Download)](https://pepy.tech/project/envpool) [![arXiv](https://img.shields.io/badge/arXiv-2206.10558-b31b1b.svg)](https://arxiv.org/abs/2206.10558) [![Read the Docs](https://img.shields.io/readthedocs/envpool)](https://envpool.readthedocs.io/) [![Unittest](https://github.com/sail-sg/envpool/actions/workflows/test.yml/badge.svg?branch=main)](https://github.com/sail-sg/envpool/actions/workflows/test.yml) [![Coverage](https://sail-sg.github.io/envpool/coverage-badge.svg)](https://sail-sg.github.io/envpool/) [![GitHub issues](https://img.shields.io/github/issues/sail-sg/envpool)](https://github.com/sail-sg/envpool/issues) [![GitHub stars](https://img.shields.io/github/stars/sail-sg/envpool)](https://github.com/sail-sg/envpool/stargazers) [![GitHub forks](https://img.shields.io/github/forks/sail-sg/envpool)](https://github.com/sail-sg/envpool/network) [![GitHub license](https://img.shields.io/github/license/sail-sg/envpool)](https://github.com/sail-sg/envpool/blob/main/LICENSE)

> **Fork status:** `1.2.5+apex.hb2` is available as a
> [GitHub wheel prerelease](https://github.com/wzhhasadream/envpool/releases/tag/wheels-1.2.5-apex.hb2-e1323c3)
> from source `e1323c3`, including native multi-task pools for every compatible
> DMC group. It is not published to PyPI. All 16 wheel build jobs passed;
> the separate comprehensive Bazel suite has Windows failures and an incomplete
> macOS run. See [validation status](#validation-status). The badges and hosted
> documentation above describe upstream EnvPool; use this repository's linked
> documentation for fork features.

**EnvPool** is a C++-based batched environment pool with pybind11 and thread pool. It has high performance (\~1M raw FPS with Atari games, \~3M raw FPS with MuJoCo simulator on DGX-A100) and compatible APIs (supports Gymnasium and dm\_env, both sync and async, both single and multi player environment). Currently it supports:

- [x] [Atari games](https://envpool.readthedocs.io/en/latest/env/atari.html)
- [x] [MuJoCo (Gymnasium)](https://envpool.readthedocs.io/en/latest/env/mujoco_gym.html)
- [x] [Classic control RL envs](https://envpool.readthedocs.io/en/latest/env/classic_control.html): CartPole, MountainCar, Pendulum, Acrobot
- [x] [Toy text RL envs](https://envpool.readthedocs.io/en/latest/env/toy_text.html): Catch, FrozenLake, Taxi, NChain, CliffWalking, Blackjack
- [x] [ViZDoom single player](https://envpool.readthedocs.io/en/latest/env/vizdoom.html)
- [x] [DeepMind Control Suite](https://envpool.readthedocs.io/en/latest/env/dm_control.html)
- [x] [Box2D](https://envpool.readthedocs.io/en/latest/env/box2d.html)
- [x] [Google Research Football](https://envpool.readthedocs.io/en/latest/env/gfootball.html)
- [x] [Procgen](https://envpool.readthedocs.io/en/latest/env/procgen.html)
- [x] [Minigrid](https://envpool.readthedocs.io/en/latest/env/minigrid.html)
- [x] [Highway](https://envpool.readthedocs.io/en/latest/env/highway.html)
- [x] [MetaWorld](https://envpool.readthedocs.io/en/latest/env/metaworld.html)
- [x] [MyoSuite](https://envpool.readthedocs.io/en/latest/env/myosuite.html)
- [x] [PGX](https://envpool.readthedocs.io/en/latest/env/pgx.html)
- [x] [Jumanji](https://envpool.readthedocs.io/en/latest/env/jumanji.html)
- [x] [MuJoCo Playground](https://envpool.readthedocs.io/en/latest/env/mujoco_playground.html)
- [x] [Native HumanoidBench](docs/env/humanoid_bench.rst): 112 registered IDs
- [x] [Native multi-task pools](docs/env/multitask.rst): compatible DMC,
  MetaWorld, MyoSuite, and Atari tasks

Here are EnvPool's several highlights:

- **All-platform support: Linux, macOS, and Windows** wheels for Python 3.11-3.14;
- **Built-in rendering** with batched `rgb_array` output and `human` display mode;
- Compatible with [`gymnasium`](https://github.com/Farama-Foundation/Gymnasium) APIs and DeepMind `dm_env` APIs;
- Manage a pool of envs, interact with the envs in batched APIs by default;
- Support both synchronous execution and asynchronous execution;
- Support both single player and multi-player environment;
- Easy C++ developer API to add new envs: [Customized C++ environment integration](https://envpool.readthedocs.io/en/latest/content/new_env.html);
- Free **\~2x** speedup with only single environment;
- **1 Million** Atari frames / **3 Million** MuJoCo steps per second simulation with 256 CPU cores, **~20x** throughput of Python subprocess-based vector env;
- **~3x** throughput of Python subprocess-based vector env on low resource setup like 12 CPU cores;
- Comparing with existing GPU-based solution ([Brax](https://github.com/google/brax) / [Isaac-gym](https://developer.nvidia.com/isaac-gym)), EnvPool is a **general** solution for various kinds of speeding-up RL environment parallelization;
- XLA support with JAX jit function: [XLA Interface](https://envpool.readthedocs.io/en/latest/content/xla_interface.html);
- Compatible with some existing RL libraries, e.g., [Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3), [Tianshou](https://github.com/thu-ml/tianshou), [ACME](https://github.com/deepmind/acme), [CleanRL](https://github.com/vwxyzjn/cleanrl), or [rl\_games](https://github.com/Denys88/rl_games).
  - Stable-Baselines3 [`Pendulum-v1` example](https://github.com/sail-sg/envpool/blob/main/examples/sb3_examples/ppo.py);
  - Tianshou [`CartPole` example](https://github.com/sail-sg/envpool/blob/main/examples/tianshou_examples/cartpole_ppo.py), [`Pendulum-v1` example](https://github.com/sail-sg/envpool/blob/main/examples/tianshou_examples/pendulum_ppo.py), [Atari example](https://github.com/thu-ml/tianshou/tree/master/examples/atari#envpool), [MuJoCo example](https://github.com/thu-ml/tianshou/tree/master/examples/mujoco#envpool), and [integration guideline](https://tianshou.readthedocs.io/en/master/tutorials/cheatsheet.html#envpool-integration);
  - ACME [`HalfCheetah` example](https://github.com/sail-sg/envpool/blob/main/examples/acme_examples/ppo_continuous.py);
  - CleanRL [`Pong-v5` example](https://github.com/sail-sg/envpool/blob/main/examples/cleanrl_examples/ppo_atari_envpool.py) ([Solving Pong in 5 mins](https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/#solving-pong-in-5-minutes-with-ppo--envpool) ([tracked experiment](https://wandb.ai/costa-huang/cleanRL/runs/opk2dmta)));
  - rl\_games [Atari example](https://github.com/Denys88/rl_games/blob/master/docs/ATARI_ENVPOOL.md) ([2 mins Pong](https://github.com/sail-sg/envpool/blob/main/demo/envpool_demo_pong.ipynb) and [15 mins Breakout](https://github.com/sail-sg/envpool/blob/main/demo/envpool_demo_breakout.ipynb)) and [MuJoCo example](https://github.com/Denys88/rl_games/blob/master/docs/MUJOCO_ENVPOOL.md) (5 mins [Ant](https://github.com/sail-sg/envpool/blob/main/demo/envpool_demo_ant.ipynb) and [HalfCheetah](https://github.com/sail-sg/envpool/blob/main/demo/envpool_demo_halfcheetah.ipynb)).

Check out our [arXiv paper](https://arxiv.org/abs/2206.10558) for more details!

## Installation

### This fork: published hb2 wheels

The [hb2 GitHub prerelease](https://github.com/wzhhasadream/envpool/releases/tag/wheels-1.2.5-apex.hb2-e1323c3)
contains 16 wheels for standard CPython 3.11-3.14 on Linux x86_64/aarch64,
macOS Apple Silicon arm64, and Windows AMD64. Download the wheel matching
your interpreter and operating system, then install that exact file. For
example, **Linux x86_64 with CPython 3.12**:

```bash
python -m pip install "https://github.com/wzhhasadream/envpool/releases/download/wheels-1.2.5-apex.hb2-e1323c3/envpool-1.2.5%2Bapex.hb2-cp312-cp312-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl"
python -c "import envpool; print(envpool.__version__)"
```

This installs `1.2.5+apex.hb2` from commit
`e1323c3f361a4c1ced4ad6f69bf4442621bca5ef`, including the expanded compatible
DMC mixing, existing multi-task backends, native HumanoidBench, and optional
Gymnasium wrappers. The release's `SHA256SUMS` lists all 16 wheel digests.
The [hb1 wheels](https://github.com/wzhhasadream/envpool/releases/tag/wheels-1.2.5-apex.hb1-1676bc9)
remain unchanged and do not contain the expanded DMC support. The distinct hb2
version avoids same-version installation ambiguity.

To build hb2 from source, use the repository's
[source-build instructions](docs/content/build.rst); installing a source
checkout with pip alone does not compile the native extensions. Do not
substitute an hb1 wheel when testing the hb2 examples below. This is a
GitHub-only prerelease; no hb2 package has been published to PyPI.

### Upstream PyPI package

Upstream EnvPool is hosted on [PyPI](https://pypi.org/project/envpool/).
An unqualified PyPI installation does not select this fork's hb2 changes.

You can simply install EnvPool with the following command:

```bash
$ pip install envpool
```

After installation, open a Python console and type

```python
import envpool
print(envpool.__version__)
```

If no error occurs, you have successfully installed EnvPool.

Platform notes:

- Linux Procgen wheels intentionally do not vendor Qt. If making a Procgen
  environment reports missing `libQt5Core.so.5` or `libQt5Gui.so.5`, install
  the system Qt 5 runtime, for example with `apt install qtbase5-dev` or
  `dnf install qt5-qtbase`.
- Windows Procgen wheels bundle the required Qt runtime DLLs (`Qt5Core.dll`
  and `Qt5Gui.dll`) next to the extension module.
- Windows source/release CI validates MuJoCo rendering with Mesa software
  OpenGL. To reproduce that setup locally, point `ENVPOOL_DLL_DIR` at a Mesa
  DLL directory and set `GALLIUM_DRIVER=llvmpipe` plus
  `MESA_GL_VERSION_OVERRIDE=4.5COMPAT`.
- Building from source still requires platform-local build dependencies,
  including Qt 5. The full per-platform setup is documented in
  [Build From Source](https://envpool.readthedocs.io/en/latest/content/build.html).

### From Source

Please refer to the [guideline](https://envpool.readthedocs.io/en/latest/content/build.html).

## Documentation

Upstream tutorials and API documentation are hosted on [envpool.readthedocs.io](https://envpool.readthedocs.io).
For this fork, start with [DMC tasks](docs/env/dm_control.rst),
[the complete multi-task compatibility groups](docs/env/multitask.rst),
[optional Gymnasium wrappers](docs/content/gymnasium_wrappers.rst), and
[known limitations](KNOWN_LIMITATIONS.md).

The example scripts are under [examples/](https://github.com/sail-sg/envpool/tree/main/examples) folder; benchmark scripts are under [benchmark/](https://github.com/sail-sg/envpool/tree/main/benchmark) folder.

## Benchmark Results

The historical benchmark tables below were produced with ALE Atari environment
`PongNoFrameskip-v4` (with environment wrappers from [OpenAI
Baselines](https://github.com/openai/baselines/blob/master/baselines/common/atari_wrappers.py))
and MuJoCo environment `Ant-v3` on different hardware setups, including a
TPUv3-8 virtual machine (VM) of 96 CPU cores and 2 NUMA nodes, and an NVIDIA
DGX-A100 of 256 CPU cores with 8 NUMA nodes. The current scripts under
[`benchmark/`](https://github.com/sail-sg/envpool/tree/main/benchmark) use
Gymnasium's `ALE/Pong-v5` and `Ant-v5`. Baselines include 1) naive Python
for-loop; 2) the most popular RL environment parallelization execution by
Python subprocess, e.g.,
[gym.vector_env](https://github.com/openai/gym/blob/master/gym/vector/vector_env.py);
3) to our knowledge, the fastest RL environment executor [Sample
Factory](https://github.com/alex-petrenko/sample-factory) before EnvPool.

We report EnvPool performance with sync mode, async mode, and NUMA + async mode, compared with the baselines on different number of workers (i.e., number of CPU cores). As we can see from the results, EnvPool achieves significant improvements over the baselines on all settings. On the high-end setup, EnvPool achieves 1 Million frames per second with Atari and 3 Million frames per second with MuJoCo on 256 CPU cores, which is 14.9x / 19.6x of the `gym.vector_env` baseline. On a typical PC setup with 12 CPU cores, EnvPool's throughput is 3.1x / 2.9x of `gym.vector_env`.

|  Atari Highest FPS   | Laptop (12) | Workstation (32) | TPU-VM (96) | DGX-A100 (256) |
| :------------------: | :---------: | :--------------: | :---------: | :------------: |
|       For-loop       |    4,893    |      7,914       |    3,993    |     4,640      |
|      Subprocess      |   15,863    |      47,699      |   46,910    |     71,943     |
|    Sample-Factory    |   28,216    |     138,847      |   222,327   |    707,494     |
|    EnvPool (sync)    |   37,396    |     133,824      |   170,380   |    427,851     |
|   EnvPool (async)    | **49,439**  |   **200,428**    |   359,559   |    891,286     |
| EnvPool (numa+async) |      /      |        /         | **373,169** | **1,069,922**  |

|  MuJoCo Highest FPS  | Laptop (12) | Workstation (32) | TPU-VM (96) | DGX-A100 (256) |
| :------------------: | :---------: | :--------------: | :---------: | :------------: |
|       For-loop       |   12,861    |      20,298      |   10,474    |     11,569     |
|      Subprocess      |   36,586    |     105,432      |   87,403    |    163,656     |
|    Sample-Factory    |   62,510    |     309,264      |   461,515   |   1,573,262    |
|    EnvPool (sync)    |   66,622    |     380,950      |   296,681   |    949,787     |
|   EnvPool (async)    | **105,126** |   **582,446**    |   887,540   |   2,363,864    |
| EnvPool (numa+async) |      /      |        /         | **896,830** | **3,134,287**  |

![](https://envpool.readthedocs.io/en/latest/_images/throughput.png)


Please refer to the [benchmark](https://envpool.readthedocs.io/en/latest/content/benchmark.html) page for more details.

## API Usage

The following content shows both synchronous and asynchronous API usage of EnvPool. You can also run the full script at [examples/env_step.py](https://github.com/sail-sg/envpool/blob/main/examples/env_step.py)

### Synchronous API

```python
import envpool
import numpy as np

# make Gymnasium env
env = envpool.make("Pong-v5", env_type="gymnasium", num_envs=100)
# or use envpool.make_gymnasium(...)
obs, info = env.reset()  # obs shape: (100, 4, 84, 84)
act = np.zeros(100, dtype=int)
obs, rew, term, trunc, info = env.step(act)
```

Under the synchronous mode, `envpool` closely resembles Gymnasium and `dm_env`. It has the `reset` and `step` functions with the same meaning. However, there is one exception in `envpool`: batch interaction is the default. Therefore, during the creation of the envpool, there is a `num_envs` argument that denotes how many envs you like to run in parallel.

```python
env = envpool.make("Pong-v5", env_type="gymnasium", num_envs=100)
```

The first dimension of `action` passed to the step function should equal `num_envs`.

```python
act = np.zeros(100, dtype=int)
```

You don't need to manually reset one environment when any of `done` is true; instead, all envs in `envpool` have enabled auto-reset by default.

### Asynchronous API

```python
import envpool
import numpy as np

# make asynchronous
num_envs = 64
batch_size = 16
env = envpool.make(
    "Pong-v5", env_type="gymnasium", num_envs=num_envs, batch_size=batch_size
)
action_num = env.single_action_space.n
env.async_reset()  # send the initial reset signal to all envs
while True:
    obs, rew, term, trunc, info = env.recv()
    env_id = info["env_id"]
    action = np.random.randint(action_num, size=batch_size)
    env.send(action, env_id)
```

In the asynchronous mode, the `step` function is split into two parts: the `send`/`recv` functions. `send` takes two arguments, a batch of action, and the corresponding `env_id` that each action should be sent to. Unlike `step`, `send` does not wait for the envs to execute and return the next state, it returns immediately after the actions are fed to the envs. (The reason why it is called async mode).

```python
env.send(action, env_id)
```
To get the "next states", we need to call the `recv` function. However, `recv` does not guarantee that you will get back the "next states" of the envs you just called `send` on. Instead, whatever envs finishes execution gets `recv`ed first.

```python
state = env.recv()
```

Besides `num_envs`, there is one more argument `batch_size`. While `num_envs` defines how many envs in total are managed by the `envpool`, `batch_size` specifies the number of envs involved each time we interact with `envpool`. e.g. There are 64 envs executing in the `envpool`, `send` and `recv` each time interacts with a batch of 16 envs.

```python
envpool.make("Pong-v5", env_type="gymnasium", num_envs=64, batch_size=16)
```

There are other configurable arguments with `envpool.make`; please check out [EnvPool Python interface introduction](https://envpool.readthedocs.io/en/latest/content/python_interface.html).

### Rendering

EnvPool exposes rendering through the Python wrapper. Create the env with
`render_mode="rgb_array"` to get batched RGB output, or
`render_mode="human"` to display a single env through OpenCV.

```python
import envpool

env = envpool.make(
    "Ant-v5",
    env_type="gymnasium",
    num_envs=4,
    render_mode="rgb_array",
    render_width=480,
    render_height=480,
)
env.reset()
frames = env.render(env_ids=[0, 2])
assert frames.shape == (2, 480, 480, 3)
```

`render()` is batch-first, so even a single render keeps the batch dimension:
`env.render().shape == (1, H, W, 3)`. If `env_ids` is omitted, EnvPool renders
`render_env_id` (default `0`). `camera_id` can be overridden per call, while
the output size is fixed at env creation time via `render_width` and
`render_height`.

The repo test suite also exercises rendering. `make bazel-test` runs repeated
render checks for every render-capable env family, and `make release-test`
includes a wheel smoke that calls `render()` after `reset()`. On Windows, the
MuJoCo render tests use the same `ENVPOOL_DLL_DIR` Mesa preload hook described
above when you want software OpenGL instead of the system driver.

```python
viewer = envpool.make(
    "WalkerWalk-v1",
    env_type="gymnasium",
    num_envs=1,
    render_mode="human",
    render_env_id=0,
)
viewer.reset()
viewer.render()
```

`render_mode="human"` returns `None` and currently supports a single env id per
call. It also requires `opencv-python` to be installed.

### Pixel Observations

For MuJoCo tasks, pixel observations can also be exposed directly through the
regular observation API by passing `from_pixels=True`. This path is produced
natively in C++, without routing through Python-side `render()`.

```python
pixels = envpool.make(
    "WalkerWalk-v1",
    env_type="gymnasium",
    num_envs=2,
    from_pixels=True,
    frame_stack=3,
    render_width=84,
    render_height=84,
)
obs, info = pixels.reset()
assert obs.shape == (2, 9, 84, 84)
```

Pixel observations use channel-first layout. With `frame_stack=1`, each
environment returns `(3, H, W)`; with `frame_stack=3`, EnvPool stacks frames on
the channel dimension and returns `(9, H, W)`. This matches the usual PyTorch
`BCHW` convention directly. If `render_width` / `render_height` are omitted,
EnvPool defaults them to `84`.

### Native Multi-Task Pools (hb2)

All 51 registered DMC tasks across 19 domains are covered by **30 compatible
state groups** and **27 pixel groups**. A group may contain just one task.
This does not mean that all 51 tasks can be mixed together: different domains,
model topologies, or meaningful state contracts are rejected, without padding.
See the [complete group listing and examples](docs/env/multitask.rst).

```python
import envpool

env = envpool.make_multitask(
    ["HumanoidStand-v1", "HumanoidWalk-v1", "HumanoidRun-v1"],
    env_type="gymnasium",
    envs_per_task=[2, 4, 2],
    seed=list(range(8)),
)
obs, info = env.reset()
obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
env.close()
```

Regular Humanoid has 21 actions. The separate CMU domain has 56 actions and
supports its own stand/walk/run pool; it cannot join regular Humanoid.
All four Manipulator tasks can share a pool while retaining their own ball,
peg, and receptacle scenes. Finger spin/turn, Fish upright/swim, and Humanoid
pure-state/feature-state combinations require `from_pixels=True`; their
different meaningful state observations cannot be mixed in state mode.

Each slot keeps its task, original model, seed stream, and task episode limit
through resets. `env.env_task_ids[env_id]` records the assigned public ID.
`info["task_id"]` is a canonical domain-local numeric ID, not the task's input
list position. DMC IDs are alphabetical by original task name, for example
Walker `run=0, stand=1, walk=2`. The field is additive for standalone DMC too.
For asynchronous batches, route actions by returned `info["env_id"]`, never
by row number or task ID. Task identity is not automatically added to the
Gymnasium observation.

### Optional Gymnasium Wrappers

Wrappers are explicit adapters; the native defaults and task logic are unchanged.
`SameStepAutoReset` provides synchronous Gymnasium same-step reset with terminal
observations in `info["final_obs"]` and terminal diagnostics in
`info["final_info"]`. `PixelObservationLayout` changes `[B, F*C, H, W]` pixels
to `[B, F, H, W, C]` without resizing or changing frame order:

```python
native = envpool.make_multitask(
    ["WalkerStand-v1", "WalkerWalk-v1", "WalkerRun-v1"],
    envs_per_task=2,
    from_pixels=True,
    frame_stack=3,
)
env = envpool.PixelObservationLayout(envpool.SameStepAutoReset(native))
obs, info = env.reset()
assert obs.shape == (6, 3, 84, 84, 3)
env.close()
```

Both wrapper orders are supported. Same-step reset requires a synchronous,
single-player pool and Gymnasium 1.1 or newer. Pixel layout alone supports
asynchronous send/receive. See [wrapper contracts](docs/content/gymnasium_wrappers.rst)
for validity masks, subset operations, and terminal success handling.

**Atari caveat:** do not combine `SameStepAutoReset` with
`episodic_life=True` if standard life-loss pseudo-resets are needed: its explicit
reset restarts the entire game on a life loss. Keep native `NEXT_STEP` reset
with optional `PixelObservationLayout`, or use `episodic_life=False` for
full-game same-step resets. This existing behavior is unchanged.

### Validation Status

The hb2 DMC change passed local Linux validation: nine Bazel targets containing
209 Python cases and 14 C++ cases, including all compatible groups, rejection
cases, standalone parity, asynchronous IDs, reset/horizon behavior, pixels,
and existing oracle/render checks. All 7,711 compared pre-existing standalone
output arrays matched the published hb1 runtime byte-for-byte. These are
bounded regression results, not universal upstream equivalence.

For release source `e1323c3`, all 16 [wheel build/install/release-test jobs](https://github.com/wzhhasadream/envpool/actions/runs/39372019218)
passed. Additional checks of the exact published Linux x86_64 CPython 3.12
wheel passed 162 DMC mixed-task cases, 9 native-wrapper cases, 8
plugin-registration cases, and `pip check`; all 17 native extension modules
were present. These wheel checks are separate from the
[comprehensive Bazel run 39372019108](https://github.com/wzhhasadream/envpool/actions/runs/39372019108):

- Linux x86_64 and ARM64 each passed 116/116 targets.
- Windows passed 109/116 targets, including all 162 DMC mixed-task cases.
  GL/render/GLFW failures and a ViZDoom pretrain timeout remain.
- macOS was cancelled at 112/116 targets with four failed targets already
  recorded. DMC mixed-task cases passed 159/162; three exact pixel-comparison
  cases failed. This is an incomplete run, not 112 passing targets.

The source's [clang-tidy run](https://github.com/wzhhasadream/envpool/actions/runs/39372019111)
also failed on string concatenation in `envpool/mujoco/dmc/multitask.h`.
These results do not establish fully green CI on every platform. See
[known limitations](KNOWN_LIMITATIONS.md) for the validation scope.

## Contributing

EnvPool is still under development. More environments will be added, and we always welcome contributions to help EnvPool better. If you would like to contribute, please check out our [contribution guideline](https://envpool.readthedocs.io/en/latest/content/contributing.html).

## License

EnvPool is under Apache2 license.

Other third-party source-code and data are under their corresponding licenses.

We do not include their source code and data in this repo.

## Citing EnvPool

If you find EnvPool useful, please cite it in your publications.

```latex
@inproceedings{weng2022envpool,
 author = {Weng, Jiayi and Lin, Min and Huang, Shengyi and Liu, Bo and Makoviichuk, Denys and Makoviychuk, Viktor and Liu, Zichen and Song, Yufan and Luo, Ting and Jiang, Yukun and Xu, Zhongwen and Yan, Shuicheng},
 booktitle = {Advances in Neural Information Processing Systems},
 editor = {S. Koyejo and S. Mohamed and A. Agarwal and D. Belgrave and K. Cho and A. Oh},
 pages = {22409--22421},
 publisher = {Curran Associates, Inc.},
 title = {Env{P}ool: A Highly Parallel Reinforcement Learning Environment Execution Engine},
 url = {https://proceedings.neurips.cc/paper_files/paper/2022/file/8caaf08e49ddbad6694fae067442ee21-Paper-Datasets_and_Benchmarks.pdf},
 volume = {35},
 year = {2022}
}
```

## Disclaimer

This is not an official Sea Limited or Garena Online Private Limited product.
