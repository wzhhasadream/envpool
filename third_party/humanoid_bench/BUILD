# SPDX-License-Identifier: Apache-2.0

load("@rules_python//python:defs.bzl", "py_library", "py_test")

package(default_visibility = ["//visibility:public"])

exports_files(["libmujoco.so.3.6.0"])

py_library(
    name = "generate_assets_build",
    srcs = ["generate_assets_build.py"],
    imports = ["../.."],
)

py_test(
    name = "generate_assets_build_test",
    srcs = ["generate_assets_build_test.py"],
    data = [
        "assets.BUILD",
        "//envpool/mujoco:humanoid_bench/registry.json",
        "@humanoid_bench_source//:source_assets",
    ],
    deps = [":generate_assets_build"],
)
