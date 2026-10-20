# Known limitations and validation scope

This page describes the current fork source, version `1.2.5+apex.hb2`.
Its expanded DMC mixing has been validated locally on Linux; hb2 wheels are
not published yet. The existing [hb1 wheel release](https://github.com/wzhhasadream/envpool/releases/tag/wheels-1.2.5-apex.hb1-1676bc9)
does not include that extension. A successful wheel install or smoke test is
not a claim of exact upstream equivalence for every task.

## Compatible pools, not arbitrary task mixing

The 51 DMC tasks form 30 state-observation groups and 27 pixel groups.
Only tasks within one compatible same-domain group may share a native pool.
Different articulated topologies and action/state contracts are not padded.
Finger, Fish, and Humanoid have legacy union observation keys that some tasks
do not populate; those keys do not establish meaningful state compatibility.
Pixel observations allow the specifically documented combinations, not all
cross-task combinations. See the [complete groups](docs/env/multitask.rst).

Task IDs are domain-local and alphabetical by original task name, not global
IDs or the request-list position. DMC adds `info["task_id"]` to single-task
results as well as mixed-task results. Consumers comparing exact info-key sets
must allow this additive field. Asynchronous actions must be routed using the
returned `env_id`.

## SameStepAutoReset with Atari episodic life

`SameStepAutoReset` performs an explicit reset after every ended transition.
With Atari `episodic_life=True`, losing a life therefore restarts the whole
game, rather than continuing the game with the remaining lives. This wrapper
behavior is unchanged in hb2.

- For native life-loss pseudo-resets, retain `NEXT_STEP` automatic reset.
  `PixelObservationLayout` may be used on its own.
- For full-game same-step resets, use `episodic_life=False` with
  `SameStepAutoReset`.

Same-step wrapping is limited to synchronous single-player pools. The pixel
layout adapter alone supports asynchronous send/receive. Both wrappers are
Python adapters and do not expose XLA. Terminal success belongs to
`final_info`, while main info on ended rows describes the reset episode.
See the [full wrapper contract](docs/content/gymnasium_wrappers.rst).

## Oracle and task semantics

- **DMC:** native reset RNG does not reproduce NumPy's random stream. Equal
  integer seeds are not a promise of identical upstream initial states.
  Same-state oracle tests synchronize the initial state and then compare
  subsequent actions without repeatedly correcting the state. Public rewards
  are float32. See [DMC compatibility notes](docs/env/dm_control.rst).
- **QuadrupedEscape:** native terrain uses independent bumps on a 201-by-201
  grid, while upstream starts with a 30-by-30 grid and cubic interpolation.
  Tests that synchronize the height field do not establish identical reset
  distributions. This existing difference is unchanged.
- **MyoSuite:** the `success` alias retains EnvPool's numeric `solved` dtype,
  unlike the upstream boolean dtype used by ApexRL. Terminal success and
  evaluation aggregates are separate concepts. Rendering checks use bounded
  visual comparisons, not a blanket claim of bitwise renderer identity.
  See [MyoSuite documentation](docs/env/myosuite.rst).
- **HumanoidBench:** flat control is supported; historical oracle comparisons
  do not establish full-episode equivalence on every platform. The upstream
  `g1-truck-v0` reward refers to a missing `free_base` joint; native code uses
  the root position. See the [historical validation scope](docs/env/humanoid_bench.rst).

## Current validation and platform limits

The hb2 DMC change passed nine local Linux Bazel targets: 209 Python cases,
including 162 multi-task cases, plus 14 C++ cases. All 7,711 compared
pre-existing standalone output arrays matched hb1 byte-for-byte. The suite
covers all registered tasks in compatible groups, rejection of incompatible
groups, per-slot seeds/horizons, action repeat, friction, resets, asynchronous
identity, pixels, and existing oracle/render regressions. Historical
unpopulated observation/debug fields are deliberately excluded from
meaningful-parity assertions; existing initialized zero fields are compared.

Fresh hb2 full-platform tests and packaged-wheel verification are pending.
The previous hb1 installed-wheel matrix passed 16 build/install/test jobs,
but its separate comprehensive Bazel run was not fully green: Windows had
graphics/render failures and a PGX timeout, and the plugin-registration test
had an independently fixed harness issue. Fixing the harness or moving a test's
Windows GL preload does not by itself prove platform-wide success.

Check the [workflows for the exact source commit](https://github.com/wzhhasadream/envpool/actions),
not only the package version. The published wheel matrix covers standard
CPython 3.11-3.14 on Linux x86_64/aarch64, macOS arm64, and Windows AMD64;
it does not include free-threaded Python, macOS Intel, or Windows ARM wheels.
