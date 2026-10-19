"""Use the active Python headers/runtime for the local Linux extension build."""

import json
import importlib.util
import shutil
import sys
import sysconfig
from pathlib import Path

destination = Path(sys.argv[1]).resolve()
destination.mkdir(parents=True, exist_ok=True)
for name, source in [("include", sysconfig.get_path("include")), ("python", sys.executable)]:
    target = destination / name
    if target.is_symlink():
        target.unlink()
    target.symlink_to(source)
(destination / "WORKSPACE").write_text('workspace(name = "python_versions_3_11_x86_64-unknown-linux-gnu")\n')
(destination / "BUILD.bazel").write_text('''load("@rules_python//python/private:local_runtime_repo_setup.bzl", "define_local_runtime_toolchain_impl")
exports_files(["python"])
define_local_runtime_toolchain_impl(
    name = "local_runtime", major = "3", minor = "11", micro = "%s",
    interpreter_path = %s, interface_library = None, libraries = [],
    implementation_name = "cpython", os = "@platforms//os:linux",
)
''' % (sys.version_info.micro, json.dumps(sys.executable)))
print(destination)
if importlib.util.find_spec("jaxlib") is not None:
    jaxlib = Path(importlib.util.find_spec("jaxlib").origin).parent
    ffi = destination.with_name("envpool-ffi")
    ffi.mkdir(parents=True, exist_ok=True)
    for header in ["api.h", "c_api.h", "ffi.h"]:
        shutil.copy2(jaxlib / "include/xla/ffi/api" / header, ffi / header)
    shutil.copy2(Path(__file__).resolve().parents[1] / "openxla_ffi/ffi_api.BUILD", ffi / "BUILD.bazel")
    (ffi / "WORKSPACE").write_text('workspace(name = "openxla_ffi_headers")\n')
