# Pinned Atari sources and test oracle

The native backend targets Farama's Arcade Learning Environment v0.11.2,
commit `72a87826916aa0b696096c991faa982992ff80ec`:
https://github.com/Farama-Foundation/Arcade-Learning-Environment/tree/72a87826916aa0b696096c991faa982992ff80ec

`generate_task_catalog.py` generates the stable task-index header from
`src/ale/python/roms/md5.json`. Its required SHA256 is
`3f1cdf2329541756248f0378ac406dacd9a0818e7abd73c13c6e8b14289cdf5e`.
The four existing multiplayer exclusions leave 104 single-player games.
Regenerate with:

```sh
python third_party/ale/generate_task_catalog.py /path/to/ale-0.11.2 \
  envpool/atari/task_catalog.h
```

Continuous alignment uses the original `ale-py==0.11.2` Gymnasium wrapper
in an isolated Python 3.12 process. `oracle_requirements.txt` pins its complete
small dependency set, including Gymnasium 1.2.3 and NumPy 2.4.3. These packages
are test-only dependencies and are not a runtime stepping bridge.

The official PyPI wheel used for local Linux x86-64 validation was
`ale_py-0.11.2-cp312-cp312-manylinux2014_x86_64.manylinux_2_17_x86_64.whl`,
SHA256 `212927f98357390e651b830835e1d24690816416d0e0d2148ad2d4c679941e1c`.
Its 108 packaged ROMs were checked against the pinned official MD5 manifest;
local builds can use these files as a ROM repository override. Both existing
`atari_roms` archive URLs were also checked: their SHA256 is
`e39e9fc379fe3f336911d928ce0a52e6ff6861258906efc5e849390867ff35f5`, all 104
supported paths are present, and all 108 ROM payloads are byte-for-byte
identical to the official wheel.

The native source uses the immutable official commit archive:
https://github.com/Farama-Foundation/Arcade-Learning-Environment/archive/72a87826916aa0b696096c991faa982992ff80ec.tar.gz

Its SHA256 is
`9e4f0dbe8c73c382a208e49c7d071edd146f5456b2a6ba8edb71bac0c2d91f05`.
On 2026-10-19, the official v0.11.2 tag was verified with `git ls-remote` to
resolve to that commit. All 706 archive files and executable modes matched
the clean official Git checkout and Git blob IDs, with no missing or extra
files. The original moving tag archive no longer matched the old repository
checksum; the immutable commit archive preserves the verified version and
source tree rather than accepting an unexplained checksum change.

The converter intentionally preserves the pinned wrapper's positive-threshold
comparisons on both sides of each axis, strict fire comparison, and radius as
paddle strength. Native float32 libm and NumPy SIMD trigonometry can differ by
one ULP at categorical thresholds. The separate boundary characterization
records that limitation without changing thresholds or applying tolerance.
See `docs/env/atari.rst` for the exact behavior, source links, and the deliberate
finite/in-Box native input-validation policy.
