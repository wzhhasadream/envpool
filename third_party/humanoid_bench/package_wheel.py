# SPDX-License-Identifier: Apache-2.0

"""Preserve the base wheel's bundled libraries in the extended binary wheel."""

import configparser
import subprocess
import sys
from pathlib import Path

from wheel.wheelfile import WheelFile

source = Path(__file__).resolve().parents[2]
libraries = source / "envpool.libs"
for name in ["humanoid_bench_envpool.so", "mujoco_dmc_envpool.so"]:
    binary = source / "envpool/mujoco" / name
    needed = subprocess.check_output(
        ["patchelf", "--print-needed", str(binary)], text=True
    ).splitlines()
    for dependency in needed:
        bundled = sorted(libraries.glob(dependency.split(".so")[0] + "-*.so*"))
        if bundled:
            subprocess.run(
                [
                    "patchelf",
                    "--replace-needed",
                    dependency,
                    bundled[0].name,
                    str(binary),
                ],
                check=True,
            )
    subprocess.run(
        ["patchelf", "--set-rpath", "$ORIGIN/../../envpool.libs", str(binary)],
        check=True,
    )
for plugin in (source / "envpool/mujoco/humanoid_bench/plugins").glob("*.so"):
    subprocess.run(
        [
            "patchelf",
            "--set-rpath",
            "$ORIGIN/../../../../envpool.libs",
            str(plugin),
        ],
        check=True,
    )

subprocess.run(
    [
        sys.executable,
        "-m",
        "pip",
        "wheel",
        "--no-deps",
        "--no-build-isolation",
        "--wheel-dir",
        sys.argv[1],
        str(source),
    ],
    check=True,
    stdout=sys.stderr,
)
config = configparser.ConfigParser()
config.read(source / "setup.cfg")
version = config["metadata"]["version"]
wheels = sorted(Path(sys.argv[1]).glob(f"envpool-{version}-*.whl"))
if len(wheels) != 1:
    raise RuntimeError(
        f"Expected one envpool {version} wheel in {sys.argv[1]}, found {wheels}"
    )
wheel = wheels[0]
destination = Path(sys.argv[2])
destination.mkdir(parents=True, exist_ok=True)
with (
    WheelFile(wheel) as original,
    WheelFile(destination / wheel.name, "w") as packed,
):
    for name in original.namelist():
        if not name.endswith(".dist-info/RECORD"):
            packed.writestr(name, original.read(name))
    for library in sorted(libraries.iterdir()):
        if library.is_file():
            packed.write(library, f"envpool.libs/{library.name}")
print(destination / wheel.name)
