Optional Gymnasium Wrappers
===========================

``envpool.SameStepAutoReset`` and ``envpool.PixelObservationLayout`` are
explicit, opt-in Python adapters. They do not change native environment
logic, configuration, or the defaults of ``make_gymnasium``. They can also
be imported from ``envpool.wrappers``.

For example:

.. code-block:: python

    import envpool
    import numpy as np

    native = envpool.make_gymnasium(
        "DogRun-v1",
        num_envs=8,
        batch_size=8,
        num_threads=4,
        from_pixels=True,
        frame_stack=3,
        action_repeat=3,
    )
    env = envpool.PixelObservationLayout(envpool.SameStepAutoReset(native))
    obs, info = env.reset()
    assert obs.shape == (8, 3, 84, 84, 3)
    assert env.single_observation_space.shape == (3, 84, 84, 3)
    actions = np.zeros_like(env.action_space.sample())
    obs, reward, terminated, truncated, info = env.step(actions)
    env.close()

The Gymnasium ``observation_space`` and ``action_space`` describe the full
``num_envs`` batch. ``single_observation_space`` and ``single_action_space``
describe one slot. For the example above, ``env.observation_space.shape`` is
``(8, 3, 84, 84, 3)``. Use ``env.action_space.sample()`` for a full-batch action,
or batch samples from ``single_action_space`` for a subset. Spaces remain
sized for ``num_envs`` when a subset or asynchronous call returns fewer rows.

Same-step automatic reset
-------------------------

``SameStepAutoReset`` converts native ``NEXT_STEP`` automatic reset to
Gymnasium's ``SAME_STEP`` contract. It requires Gymnasium 1.1 or newer; the info
format is verified against the pinned Gymnasium 1.2.3 implementation.

When a slot terminates or truncates, the same ``step`` call resets that slot:

* ``reward``, ``terminated``, and ``truncated`` describe the terminal transition.
* The main observation and main info for that slot describe the new episode.
* ``info["final_obs"]`` is an object array containing terminal observations.
  ``info["_final_obs"]`` marks its valid rows; other entries are ``None``.
* ``info["final_info"]`` is a recursive dictionary of batched terminal fields,
  including ``task_id`` and ``success`` when the environment exposes them.
  ``info["_final_info"]`` marks terminal rows. Every nested field also has its
  Gymnasium-style validity mask, such as ``final_info["_success"]``.

For example, terminal success is
``info["final_info"]["success"][info["_final_info"]]``. Main ``success``
belongs to the reset episode for ended rows; do not use it as terminal success.
Episode statistics are kept with the episode that produced them, including
when reset and step have different info keys. Read a field only where its
validity mask is true. Terminal data is copied before native reset can reuse
any receive buffer, including buffers for continuing slots and rewards.

Only synchronous single-player pools are supported: the effective batch size
must equal ``num_envs`` and ``max_num_players`` must be 1. Construction rejects
asynchronous or multiplayer pools. ``send``, ``recv``, ``async_reset``, and
``xla`` raise ``NotImplementedError`` before submitting work. Use only the
wrapped pool's ``step`` and ``reset`` once it is wrapped; do not interleave
operations on its underlying pool.

Synchronous subsets are supported through ``step(action, env_id)``,
``reset(env_id)``, and ``reset(options={"reset_mask": mask})``. Returned arrays
and masks have the selected batch size and follow the returned
``info["env_id"]`` order, not necessarily global slot order. Internal resets
are routed and merged by their returned IDs. Duplicate or invalid explicit
slot IDs are rejected.

Pixel observation layout
------------------------

``PixelObservationLayout`` changes native pixels from ``[B, F*C, H, W]`` to
``[B, F, H, W, C]``. The corresponding single observation space changes from
``[F*C, H, W]`` to ``[F, H, W, C]`` with exactly transformed bounds and dtype.
The batched observation space adds the leading ``num_envs`` dimension.
It does not resize, normalize, recolor, or reorder the frame history.

* MuJoCo pixels use ``frame_stack`` for F and RGB's C=3.
* Atari uses ``stack_num`` for F and ``gray_scale`` for C=1 or C=3.
  Gray-scale images retain the trailing channel dimension even for a single
  frame.
* Dictionary observations are supported recursively. Three-dimensional uint8
  pixel Box fields are transformed, and non-pixel fields remain unchanged.
* Reset, step, and masked ``final_obs`` observations are all transformed.
  Both wrapper orders are supported.
* Pixel-only wrapping supports asynchronous ``send``/``recv`` and
  ``async_reset`` without changing ``env_id`` routing. Combining it with
  ``SameStepAutoReset`` retains that adapter's synchronous-only restriction.
* Both wrappers forward ``render`` unchanged, including ``env_ids`` and
  ``camera_id``, and preserve rendering and automatic reset metadata.

Pixel transposes may be views; call ``np.ascontiguousarray`` if a downstream
consumer requires contiguous storage. Neither Python adapter supports XLA;
use the native pool directly when a compiled step API is required.
