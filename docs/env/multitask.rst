Native Multi-Task Pools
=======================

``envpool.make_multitask`` runs compatible tasks from one environment family
inside one native EnvPool. Every environment slot has a fixed task. All slots
share the existing C++ worker queue; the Python constructor only resolves task
names, counts, and configuration.

Supported Families
------------------

* MetaWorld: all 50 v3 tasks share the same observation and action contract.
* DMC Dog: ``DogRun-v1``, ``DogStand-v1``, ``DogTrot-v1``, and ``DogWalk-v1``
  can be mixed. ``DogFetch-v1`` cannot be mixed with locomotion tasks because
  its model topology differs.
* MyoSuite: tasks must have matching observation and action dimensions and
  the complete state/action schema. Different-size tasks are rejected, even
  when pixel observations are requested.
* Atari: discrete pools require the canonical 18-action ALE space. The
  constructor enables ``full_action_space=True`` and rejects an explicit
  ``False`` for discrete pools. ``LostLuggage-v5`` and ``Skiing-v5`` have nine
  legal actions and cannot join a discrete mixed pool. Continuous pools
  (``continuous=True``) use ALE's shared three-component float action and
  support all 104 games, including these two. Continuous mode accepts either
  value of ``full_action_space``, matching ALE. Ordinary single-game ``make``
  retains its existing discrete minimal-action default.

Tasks from different native backends cannot be mixed. The constructor checks
all state and action keys, shapes, dtypes, bounds, and non-task configuration.
It does not silently pad observations or actions.

Construction and Slot Assignment
--------------------------------

.. code-block:: python

    import envpool

    env = envpool.make_multitask(
        ["DogRun-v1", "DogStand-v1", "DogTrot-v1"],
        env_type="gymnasium",
        envs_per_task=8,
        seed=42,
    )

This creates 24 environments: IDs 0--7 run, 8--15 stand, and 16--23 trot.
``envs_per_task=[8, 4, 12]`` assigns a different positive count to each task.
If ``num_envs`` is supplied, it must equal the sum of these counts.
``env.task_ids`` records the requested public task IDs;
``env.env_task_ids[env_id]`` records the public task assigned to a slot.
Duplicate task IDs are allowed, so a list with one task per slot and the
(default) count of one also supports arbitrary fixed ordering.

The returned object is the family's usual native pool, supporting ``step``,
``send``, ``recv``, ``reset``, rendering, and ``env_type="dm"``. Existing
single-task ``make`` calls are unchanged.

``info["task_id"]`` is the family's canonical numeric task ID, not the task's
position in the supplied list. MetaWorld and MyoSuite preserve their existing
IDs. Dog uses ``fetch=0, run=1, stand=2, trot=3, walk=4``. These IDs are local to
a family. The Gymnasium observation does not automatically include task
identity; a task-conditioned policy should consume it explicitly.

Continuous Atari Actions
------------------------

``continuous=True`` selects native continuous-action bindings in ``make``,
``make_spec``, and ``make_multitask``. The observation contract is unchanged.
Actions are ``float32`` vectors of magnitude, angle in radians, and fire, with
bounds ``[0, -pi, 0]`` to ``[1, pi, 1]``. The native conversion follows the
pinned ALE implementation; no Python action wrapper is used.
``continuous_action_threshold`` retains ALE's default of ``0.5``.

.. code-block:: python

    env = envpool.make_multitask(
        ["Pong-v5", "Skiing-v5", "LostLuggage-v5"],
        continuous=True,
        envs_per_task=4,
        seed=42,
    )

A pool has one action type: discrete and continuous action contracts cannot
be mixed within the same pool.

Seeds and Episode Boundaries
----------------------------

Each slot keeps its task through explicit resets, subset resets, and automatic
episode resets. Tasks are not resampled at reset. Each environment initializes
its own model, task metadata, and random-number generator.

An integer seed uses the existing ``seed + env_id`` convention. A seed list
(or ``env_seed``) supplies one seed per environment slot, in slot order. Use
explicit per-slot seeds when comparing with standalone tasks or rearranging
slots. Reset continues that slot's random stream, as in ordinary EnvPool.

Each task retains its registered episode limit. A supplied
``max_episode_steps`` overrides the limit for all selected tasks. Other
configuration is pool-wide, including MuJoCo's ``action_repeat`` and
``frame_stack``, Atari's ``frame_skip`` and ``stack_num``, and image resolution.
Task-specific native settings, such as MyoSuite's
model and control timestep, still come from the assigned task's metadata.

Asynchronous Batches
--------------------

With ``batch_size`` smaller than the total slot count, returned rows can arrive
in a different order. Always use returned environment IDs when sending the
next actions:

.. code-block:: python

    env = envpool.make_multitask(
        ["MetaWorld/Reach-v3", "MetaWorld/Push-v3"],
        envs_per_task=8,
        batch_size=8,
        seed=42,
    )
    env.async_reset()
    obs, reward, terminated, truncated, info = env.recv()
    action = policy(obs, info["task_id"])
    env.send(action, info["env_id"])

Only one action or reset may be in flight for a given environment. Fixed task
slot counts do not guarantee equal task counts in asynchronous samples: faster
tasks can finish more often. Full synchronous batches preserve the configured
slot proportions on every step. Subset ``reset`` follows the existing
synchronous API; ``reset`` is not defined for asynchronous mode.

Pixels and Action Repetition
----------------------------

.. code-block:: python

    env = envpool.make_multitask(
        ["MetaWorld/Reach-v3", "MetaWorld/Push-v3"],
        envs_per_task=[4, 4],
        from_pixels=True,
        frame_stack=3,
        action_repeat=2,
        seed=list(range(8)),
    )

Pixel observations have the same channel-first, channel-stacked contract as
ordinary MuJoCo EnvPool tasks. Action repetition uses each environment's own
episode limit and preserves task assignment. See :doc:`mujoco_action_repeat`
for reward aggregation and terminal-step behavior.
