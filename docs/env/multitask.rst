Native Multi-Task Pools
=======================

``envpool.make_multitask`` runs compatible tasks from one environment family
inside one native EnvPool. Every environment slot has a fixed task. All slots
share the existing C++ worker queue; the Python constructor only resolves task
names, counts, and configuration.

The comprehensive DMC groups on this page require the current
``1.2.5+apex.hb2`` source. Published ``hb1`` wheels retain their earlier Dog-only
DMC mixing; ``hb2`` wheels have not been published yet. Source validation and
wheel availability are separate. See :doc:`humanoid_bench` for the existing
``hb1`` download and :doc:`../content/build` for native source builds.

Supported Families
------------------

* MetaWorld: all 50 v3 tasks share the same observation and action contract.
* DMC: all 51 registered tasks can be used in native pools. Mix tasks from
  one domain only within the state or pixel compatibility groups below.
  Single-task groups support homogeneous pools, not arbitrary cross-domain
  mixing.
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
all state and action keys, shapes, data types, bounds, and non-task configuration.
It does not silently pad observations or actions.

DMC Compatibility Groups
------------------------

The complete state-observation groups below cover all 51 DMC tasks. Each
semicolon separates incompatible groups within a domain. A group containing
one ID is valid on its own, including multiple slots of that same task.
Tasks from different domains never share a pool, even if some dimensions
happen to match.

* Acrobot: ``AcrobotSwingup-v1``, ``AcrobotSwingupSparse-v1``.
* Ball-in-cup: ``BallInCupCatch-v1``.
* Cartpole: ``CartpoleBalance-v1``, ``CartpoleBalanceSparse-v1``,
  ``CartpoleSwingup-v1``, ``CartpoleSwingupSparse-v1``;
  ``CartpoleThreePoles-v1``; ``CartpoleTwoPoles-v1``.
* Cheetah: ``CheetahRun-v1``.
* Dog: ``DogRun-v1``, ``DogStand-v1``, ``DogTrot-v1``, ``DogWalk-v1``;
  ``DogFetch-v1``.
* Finger: ``FingerTurnEasy-v1``, ``FingerTurnHard-v1``; ``FingerSpin-v1``.
* Fish: ``FishSwim-v1``; ``FishUpright-v1``.
* Hopper: ``HopperHop-v1``, ``HopperStand-v1``.
* Humanoid: ``HumanoidRun-v1``, ``HumanoidStand-v1``, ``HumanoidWalk-v1``;
  ``HumanoidRunPureState-v1``.
* Humanoid CMU: ``HumanoidCMURun-v1``, ``HumanoidCMUStand-v1``,
  ``HumanoidCMUWalk-v1``.
* LQR: ``LqrLqr21-v1``; ``LqrLqr62-v1``.
* Manipulator: ``ManipulatorBringBall-v1``, ``ManipulatorBringPeg-v1``,
  ``ManipulatorInsertBall-v1``, ``ManipulatorInsertPeg-v1``.
* Pendulum: ``PendulumSwingup-v1``.
* Point-mass: ``PointMassEasy-v1``, ``PointMassHard-v1``.
* Quadruped: ``QuadrupedRun-v1``, ``QuadrupedWalk-v1``;
  ``QuadrupedEscape-v1``; ``QuadrupedFetch-v1``.
* Reacher: ``ReacherEasy-v1``, ``ReacherHard-v1``.
* Stacker: ``StackerStack2-v1``; ``StackerStack4-v1``.
* Swimmer: ``SwimmerSwimmer15-v1``; ``SwimmerSwimmer6-v1``.
* Walker: ``WalkerRun-v1``, ``WalkerStand-v1``, ``WalkerWalk-v1``.

With ``from_pixels=True``, three domains additionally allow these larger
groups because the common pixel observation replaces their different state
observation semantics while retaining compatible articulated topology and
actions:

* Finger: ``FingerSpin-v1``, ``FingerTurnEasy-v1``, ``FingerTurnHard-v1``.
* Fish: ``FishSwim-v1``, ``FishUpright-v1``.
* Humanoid: ``HumanoidRun-v1``, ``HumanoidRunPureState-v1``,
  ``HumanoidStand-v1``, ``HumanoidWalk-v1``.

All other group boundaries still apply with pixels. Pixels do not permit
padding or mixing different pole, box, or swimmer-link counts, different LQR
dimensions, Dog fetch with locomotion, or Quadruped escape/fetch with
locomotion. Regular Humanoid uses 21 actions and Humanoid CMU uses 56: they
are different domains and cannot be mixed in state or pixel mode.

Compatibility includes observation meaning, not just declared array shapes.
Finger spin does not provide the turns' target state; Fish upright does not
provide the swim target; Humanoid pure-state and feature-state observations
represent different state contracts. Their historical extra observation keys
are not permission to mix these state groups or introduce arbitrary padding.

Manipulator's four tasks do share arm, object-pose, object-velocity, target,
and action semantics. Each slot loads its original ball or peg scene, with
its task's receptacle when required. Likewise, Quadruped run and walk retain
their own floor extents. Compatibility does not mean replacing all slots'
models with the first task's model.

See :doc:`dm_control` for DMC construction examples, standalone task specs,
and rendering conventions.

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
single-task ``make`` behavior is preserved. DMC adds ``info["task_id"]`` to
single-task as well as multi-task results; callers comparing exact info-key
sets should allow this additive field.

``info["task_id"]`` is the family's canonical numeric task ID, not the task's
position in the supplied list. MetaWorld and MyoSuite preserve their existing
IDs. DMC IDs are zero-based and alphabetical by the original task name
within each domain, independent of selected tasks, duplicate entries, and
slot order. For example, Dog uses
``fetch=0, run=1, stand=2, trot=3, walk=4``; Walker uses
``run=0, stand=1, walk=2``; Swimmer uses ``swimmer15=0, swimmer6=1``
(lexicographic order, not numeric link-count order). These IDs are
domain-local, not globally unique across DMC. The Gymnasium observation does
not automatically include task identity; a task-conditioned policy should
consume it explicitly.

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
DMC keeps the assigned task's model, reset distribution, reward, termination,
and domain camera default. Its episode limit counts original control steps,
including when ``action_repeat`` stops partway through a repeat window.
Task-specific native settings, such as MyoSuite's model and control timestep,
still come from the assigned task's metadata. No Python task-switching or
stepping wrapper is introduced; existing wrappers keep their usual behavior.

For Atari, combining ``SameStepAutoReset`` with ``episodic_life=True`` forces
a full-game reset on life loss. Keep native ``NEXT_STEP`` behavior for standard
life-loss resets, or set ``episodic_life=False`` for full-game same-step resets.
See :doc:`../content/gymnasium_wrappers` for this unchanged limitation and
terminal-information handling.

Asynchronous Batches
--------------------

With ``batch_size`` smaller than the total slot count, returned rows can arrive
in a different order. Always use returned environment IDs when sending the
next actions. ``info["env_id"]`` identifies the persistent slot and routes
actions; ``info["task_id"]`` identifies that row's task. Several slots can
have the same task ID, and a returned row number is not a slot ID:

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
