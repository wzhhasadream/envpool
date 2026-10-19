# Copyright 2022 Garena Online Private Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

load("@pybind11_bazel//:build_defs.bzl", "pybind_extension")
load("@python_versions//3.12:defs.bzl", py_binary_312 = "py_binary")
load("@rules_cc//cc:defs.bzl", "cc_library", "cc_test")
load("@rules_python//python:defs.bzl", "py_library", "py_test")
load("//envpool:requirements.bzl", "requirement")
load("//third_party:common.bzl", "copy_to_directory")
load("//third_party/ale:oracle_requirements.bzl", "oracle_requirement")

package(default_visibility = ["//visibility:public"])

copy_to_directory(
    name = "gen_atari_roms",
    srcs = ["@atari_roms//:roms"],
    out = "roms",
    flatten = True,
    strip_prefix = "ROM/",
)

genrule(
    name = "gen_pretrain_weight",
    srcs = [
        "@pretrain_weight//:pth",
    ],
    outs = [
        "policy-pong.pth",
        "policy-breakout.pth",
    ],
    cmd = "cp -r $(SRCS) $(@D)",
)

py_library(
    name = "atari",
    srcs = ["__init__.py"],
    data = [
        ":atari_envpool",
        ":gen_atari_roms",
    ],
    imports = ["../.."],
    deps = ["//envpool/python:api"],
)

cc_library(
    name = "atari_env",
    hdrs = [
        "atari_env.h",
        "continuous_action.h",
        "task_catalog.h",
    ],
    data = [
        ":gen_atari_roms",
    ],
    deps = [
        "//envpool/core:async_envpool",
        "//envpool/utils:image_process",
        "@ale//:ale_interface",
    ],
)

cc_test(
    name = "atari_env_test",
    size = "enormous",
    srcs = ["atari_env_test.cc"],
    deps = [
        ":atari_env",
        "//envpool/core:logging",
        "@com_google_googletest//:gtest_main",
    ],
)

py_library(
    name = "atari_network",
    srcs = ["atari_network.py"],
    imports = ["../.."],
    deps = [
        requirement("numpy"),
        requirement("torch"),
    ],
)

pybind_extension(
    name = "atari_envpool",
    srcs = [
        "atari_envpool.cc",
    ],
    linkopts = select({
        "//:windows": [],
        "//conditions:default": [
            "-ldl",
        ],
    }),
    deps = [
        ":atari_env",
        "//envpool/core:py_envpool",
    ],
)

py_library(
    name = "atari_registration",
    srcs = ["registration.py"],
    imports = ["../.."],
    deps = [
        "//envpool:registration",
    ],
)

py_test(
    name = "api_test",
    srcs = ["api_test.py"],
    imports = ["../.."],
    deps = [
        ":atari",
        ":atari_registration",
        "//envpool/python",
        requirement("numpy"),
    ],
)

py_test(
    name = "atari_render_test",
    size = "enormous",
    srcs = ["atari_render_test.py"],
    imports = ["../.."],
    deps = [
        ":atari",
        ":atari_registration",
        requirement("numpy"),
        requirement("absl-py"),
    ],
)

py_test(
    name = "atari_envpool_test",
    size = "enormous",
    srcs = ["atari_envpool_test.py"],
    imports = ["../.."],
    deps = [
        ":atari",
        ":atari_registration",
        requirement("numpy"),
        requirement("jax"),
        requirement("dm-env"),
        requirement("absl-py"),
        requirement("opencv-python-headless"),
    ],
)

py_test(
    name = "atari_pretrain_test",
    size = "enormous",
    srcs = ["atari_pretrain_test.py"],
    data = [
        ":gen_pretrain_weight",
    ],
    imports = ["../.."],
    deps = [
        ":atari",
        ":atari_network",
        ":atari_registration",
        requirement("numpy"),
        requirement("absl-py"),
        requirement("tianshou"),
        requirement("tqdm"),
        requirement("torch"),
    ],
)

py_test(
    name = "atari_multitask_test",
    size = "enormous",
    srcs = ["atari_multitask_test.py"],
    imports = ["../.."],
    deps = [
        ":atari",
        ":atari_registration",
        requirement("numpy"),
        requirement("absl-py"),
    ],
)

py_test(
    name = "atari_continuous_test",
    size = "enormous",
    srcs = ["atari_continuous_test.py"],
    imports = ["../.."],
    deps = [
        ":atari",
        ":atari_registration",
        requirement("numpy"),
        requirement("absl-py"),
    ],
)

py_binary_312(
    name = "atari_continuous_oracle_probe",
    srcs = ["atari_continuous_oracle_probe.py"],
    deps = [
        oracle_requirement("ale-py"),
        oracle_requirement("gymnasium"),
        oracle_requirement("numpy"),
    ],
)

py_test(
    name = "atari_continuous_align_test",
    size = "enormous",
    srcs = ["atari_continuous_align_test.py"],
    data = [":atari_continuous_oracle_probe"],
    imports = ["../.."],
    deps = [
        ":atari",
        ":atari_registration",
        requirement("numpy"),
        requirement("absl-py"),
    ],
)

py_test(
    name = "continuous_api_test",
    srcs = ["continuous_api_test.py"],
    imports = ["../.."],
    deps = [
        ":atari",
        ":atari_registration",
        requirement("numpy"),
        requirement("absl-py"),
    ],
)

py_test(
    name = "atari_continuous_xla_test",
    srcs = ["atari_continuous_xla_test.py"],
    imports = ["../.."],
    deps = [
        ":atari",
        ":atari_registration",
        requirement("numpy"),
        requirement("jax"),
        requirement("absl-py"),
    ],
)
