Native MuJoCo Action Repeat
===========================

Gym MuJoCo, DeepMind Control Suite, MyoSuite, and MetaWorld accept
``action_repeat=N`` in ``envpool.make_gymnasium``, ``envpool.make_dm``, and
``envpool.make_spec``. This follows the native HumanoidBench convention.
``N`` must be a positive integer and defaults to 1. Both state and pixel
observation variants support it; no Python stepping wrapper is needed.

.. code-block:: python

   import envpool
   import numpy as np

   env = envpool.make_gymnasium(
       "HalfCheetah-v4", num_envs=16, seed=0,
       action_repeat=3, frame_stack=2,
   )
   obs, info = env.reset()
   action = np.zeros((16, 6), dtype=np.float64)
   obs, reward, terminated, truncated, info = env.step(action)
   # reward: sum of up to 3 original control-step rewards
   # info["per_timestep_reward"]: reward of the final control step
   # info["elapsed_step"]: original control steps since reset
   env.close()

Stepping and rewards
--------------------

Each external ``step`` repeats the supplied action for up to ``N`` complete
original environment control steps, on the C++ worker. Every control step
runs its original physics, action transformations, task hooks, reward, and
termination checks. Repetition stops as soon as the environment terminates or
reaches its time limit. It never automatically starts another episode inside
the same repeated step. The next call retains EnvPool's existing next-step
reset behavior.

The public float32 reward is the sum of the executed control-step rewards,
without discounting. Accumulation uses double precision.
``info["per_timestep_reward"]`` is a per-environment scalar containing only
the final control-step reward (float32 for Gym and DMC, float64 for MyoSuite
and MetaWorld). Other reward components, success flags, and task diagnostics
also describe the final control step. They are not sums over the repetition
window.

With the default repeat of 1, existing step timing and diagnostic meanings
are unchanged. The new ``per_timestep_reward`` field is additive; clients
that compare the exact set of info keys should allow this extra field.

``max_episode_steps`` and ``info["elapsed_step"]`` count original control
steps. For example, a limit of 5 with repeat 3 produces elapsed counts 3 and
5, with truncation on the second call. The next call resets to elapsed 0.
Existing family-specific ``dm_env`` discount semantics are preserved:
DMC time limits retain discount 1, and DMC task termination gives discount 0;
Gym, MyoSuite, and MetaWorld retain their existing discount 0 on done.
DMC reset rewards are now explicitly zero, also fixing previously stale or
uninitialized reset rewards.

Physics and observations
------------------------

``action_repeat`` does not change ``frame_skip`` or the simulation timestep.
Away from an episode boundary, one external step advances
``N * frame_skip`` physics steps, while reward calculations still use the
original control timestep. Default ``frame_skip`` values remain:

* Gym: 2 for Reacher and InvertedPendulum; 4 for Hopper, Swimmer, and
  Walker2d; 5 for Ant, HalfCheetah, Humanoid, HumanoidStandup,
  InvertedDoublePendulum, and Pusher.
* DMC: 1 for Acrobot, Cartpole, Cheetah, LQR, Pendulum, PointMass, and Reacher;
  2 for Finger; 3 for Dog; 4 for Hopper and Quadruped; 5 for Humanoid;
  10 for BallInCup, Fish, HumanoidCMU, Manipulator, Stacker, and Walker;
  15 for Swimmer.
* MetaWorld: 5.
* MyoSuite: the original generated per-task value (5 or 10), also exposed
  in ``info["frame_skip"]`` in test/debug builds. There is no new
  family-level ``frame_skip`` override.

Only the final state observation or pixel frame is published. ``frame_stack``
advances once per external call, stacking external observations. Internal task
history still advances every control step: MetaWorld's intrinsic previous
18-dimensional observation is the previous control step, and MyoSuite's
fatigue, controllers, and task state evolve at their original rate.
MetaWorld's relative hand displacement is reapplied for every control step.

Validation
----------

``//envpool/mujoco:action_repeat_test`` compares native repetition with manual
repeat-1 stepping for every registered task, and validates all aliases' specs.
It covers default equivalence, invalid repeats, summed rewards, final-step
information, frame stacks, partial repeats at time limits, early termination,
next-step reset behavior, and ``dm_env`` discounts. Existing family oracle
alignment tests continue to validate the underlying single control step.
