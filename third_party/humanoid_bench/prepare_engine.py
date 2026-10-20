# SPDX-License-Identifier: Apache-2.0

"""Bundle the official Linux MuJoCo 3.6.0 engine and its model decoders."""

import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

root = Path(__file__).resolve().parents[2]
package = root / "envpool/mujoco/humanoid_bench"
plugins = package / "plugins"
plugins.mkdir(parents=True, exist_ok=True)
if len(sys.argv) > 1:
    source = Path(sys.argv[1])
else:
    cache = root / "build/mujoco-official"
    cache.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "download",
            "--no-deps",
            "--only-binary=:all:",
            "mujoco==3.6.0",
            "--dest",
            str(cache),
        ],
        check=True,
    )
    with zipfile.ZipFile(next(cache.glob("mujoco-3.6.0-*.whl"))) as wheel:
        wheel.extractall(cache / "unpacked")
    source = cache / "unpacked/mujoco"
engine = source / "libmujoco.so.3.6.0"
shutil.copy2(engine, Path(__file__).with_name(engine.name))
libraries = root / "envpool.libs"
libraries.mkdir(exist_ok=True)
shutil.copy2(engine, libraries / engine.name)
for name in ["libobj_decoder.so", "libstl_decoder.so", "libsensor.so"]:
    shutil.copy2(source / "plugin" / name, plugins / name)
    subprocess.run(
        [
            "patchelf",
            "--set-rpath",
            "$ORIGIN/../../../../envpool.libs",
            str(plugins / name),
        ],
        check=True,
    )
licenses = next(source.parent.glob("mujoco-3.6.0.dist-info/licenses"))
shutil.copytree(licenses, package / "mujoco_licenses", dirs_exist_ok=True)
print(engine)
