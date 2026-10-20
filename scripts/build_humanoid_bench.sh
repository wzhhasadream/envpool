#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python="${PYTHON:-python}"
build_root="${BUILD_ROOT:-$root/build}"
bazel="${BAZEL:-$build_root/tools/bazel}"
cd "$root"

"$python" -c 'import sys; assert sys.platform == "linux" and sys.version_info[:2] == (3, 11), "Use Linux with CPython 3.11"'
mkdir -p "$build_root/tools" "$build_root/base-wheel"
if [[ ! -x "$bazel" ]]; then
    curl -fL --retry 3 -o "$bazel" https://releases.bazel.build/8.6.0/release/bazel-8.6.0-linux-x86_64
    chmod +x "$bazel"
fi

# Preserve the released native families while rebuilding HumanoidBench and DMC.
"$python" -m pip download --no-deps --only-binary=:all: envpool==1.2.5 --dest "$build_root/base-wheel"
"$python" - "$root" "$build_root/base-wheel" <<'PY'
import sys
import zipfile
from pathlib import Path

root, cache = map(Path, sys.argv[1:])
with zipfile.ZipFile(next(cache.glob("envpool-1.2.5-*.whl"))) as wheel:
    for entry in wheel.infolist():
        if entry.filename.startswith(("envpool/", "envpool.libs/")) and not entry.filename.endswith(("/", ".py", ".pyc")):
            wheel.extract(entry, root)
            (root / entry.filename).chmod((entry.external_attr >> 16) & 0o777)
PY
"$python" third_party/humanoid_bench/prepare_assets.py
"$python" third_party/humanoid_bench/prepare_engine.py
"$python" third_party/humanoid_bench/local_python.py "$build_root/envpool-python"
ffi_options=()
if [[ -f "$build_root/envpool-ffi/BUILD.bazel" ]]; then
    ffi_options=(--override_repository="openxla_ffi_headers=$build_root/envpool-ffi")
fi
cp third_party/pip_requirements/requirements-release-lock.txt third_party/pip_requirements/requirements.txt

"$bazel" --output_user_root="$build_root/bazel" build --config=release --define=envpool_official_mujoco=true --jobs="${JOBS:-1}" --per_file_copt='envpool/mujoco/dmc/mujoco_envpool.cc@-O1' --override_repository="python_versions_3_11_x86_64-unknown-linux-gnu=$build_root/envpool-python" "${ffi_options[@]}" --@rules_python//python/config_settings:python_version=3.11 //envpool/mujoco:humanoid_bench_envpool //envpool/mujoco:mujoco_dmc_envpool
cp bazel-bin/envpool/mujoco/humanoid_bench_envpool.so envpool/mujoco/
cp bazel-bin/envpool/mujoco/mujoco_dmc_envpool.so envpool/mujoco/
wheel="$("$python" third_party/humanoid_bench/package_wheel.py "$build_root/envpool-wheel" "$root/dist")"
"$python" -m pip install --no-deps --force-reinstall "$wheel"
