# Copyright 2026 EnvPool Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Regression tests for the complete HumanoidBench asset closure."""

import os
import tempfile
import unittest
from pathlib import Path

from third_party.humanoid_bench.generate_assets_build import (
    build_file,
    runtime_assets,
)


class AssetClosureTest(unittest.TestCase):
    """Keep every registered model resource and reject incomplete closures."""

    def setUp(self) -> None:
        """Create an isolated set of synthetic model resources."""
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)

    def write(self, path: str, content: str = "asset") -> None:
        """Write one synthetic asset under the temporary root."""
        target = self.root / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content)

    def test_every_root_nested_include_and_license(self) -> None:
        """Include every task and license, excluding only unreachable data."""
        self.write(
            "envs/a.xml", '<mujoco><include file="../parts/a.xml"/></mujoco>'
        )
        self.write(
            "envs/b.xml", '<mujoco><include file="../parts/b.xml"/></mujoco>'
        )
        self.write(
            "parts/a.xml", '<mujoco><include file="nested/a.xml"/></mujoco>'
        )
        self.write(
            "parts/b.xml", '<mujoco><hfield file="terrain.bin"/></mujoco>'
        )
        self.write(
            "parts/nested/a.xml", '<mujoco><mesh file="shape.obj"/></mujoco>'
        )
        self.write("parts/nested/shape.obj")
        self.write("parts/terrain.bin")
        self.write("unused/model.xml", "deliberately invalid XML")
        self.write("unused/LICENSE")
        self.write("unused/NOTICE.txt")
        self.assertEqual(
            runtime_assets(self.root, ["envs/a.xml", "envs/b.xml"]),
            [
                "envs/a.xml",
                "envs/b.xml",
                "parts/a.xml",
                "parts/b.xml",
                "parts/nested/a.xml",
                "parts/nested/shape.obj",
                "parts/terrain.bin",
                "unused/LICENSE",
                "unused/NOTICE.txt",
            ],
        )

    def test_model_directory_takes_precedence(self) -> None:
        """Preserve MuJoCo's legacy model-relative resolution order."""
        self.write(
            "envs/a.xml", '<mujoco><include file="../parts/a.xml"/></mujoco>'
        )
        self.write("parts/a.xml", '<mujoco><mesh file="shape.obj"/></mujoco>')
        self.write("envs/shape.obj")
        self.write("parts/shape.obj")
        self.assertEqual(
            runtime_assets(self.root, ["envs/a.xml"]),
            ["envs/a.xml", "envs/shape.obj", "parts/a.xml"],
        )

    def test_compiler_directories_and_texture_faces(self) -> None:
        """Respect compiler prefixes for meshes, textures, and heightfields."""
        self.write(
            "a.xml",
            '<mujoco><compiler assetdir="assets" meshdir="meshes" '
            'texturedir="textures"/><asset><mesh file="shape.obj"/>'
            '<skin file="skin.skn"/><hfield file="height.bin"/>'
            '<texture fileright="right.png" fileleft="left.png"/>'
            "</asset></mujoco>",
        )
        paths = [
            "a.xml",
            "assets/height.bin",
            "meshes/shape.obj",
            "meshes/skin.skn",
            "textures/left.png",
            "textures/right.png",
        ]
        for path in paths[1:]:
            self.write(path)
        self.assertEqual(runtime_assets(self.root, ["a.xml"]), paths)

    def test_strip_path(self) -> None:
        """Apply strippath before adding the compiler's mesh directory."""
        self.write(
            "a.xml",
            '<mujoco><compiler strippath="true" meshdir="meshes"/>'
            '<asset><mesh file="unused/shape.obj"/></asset></mujoco>',
        )
        self.write("meshes/shape.obj")
        self.assertEqual(
            runtime_assets(self.root, ["a.xml"]), ["a.xml", "meshes/shape.obj"]
        )

    def test_missing_asset_fails(self) -> None:
        """Never silently prune an unresolved runtime reference."""
        self.write(
            "a.xml",
            '<mujoco><asset><mesh file="missing.obj"/></asset></mujoco>',
        )
        with self.assertRaises(FileNotFoundError):
            runtime_assets(self.root, ["a.xml"])

    def test_absolute_and_escaping_paths_fail(self) -> None:
        """Require every resource to fit inside the relocatable asset tree."""
        for path in ("/unpackaged.obj", "../unpackaged.obj"):
            with self.subTest(path=path):
                self.write(
                    "a.xml",
                    f'<mujoco><asset><mesh file="{path}"/></asset></mujoco>',
                )
                with self.assertRaises(ValueError):
                    runtime_assets(self.root, ["a.xml"])

    def test_unknown_resource_fails(self) -> None:
        """Fail closed if a future model adds an unsupported file reference."""
        self.write("a.xml", '<mujoco><unrecognized file="shape.obj"/></mujoco>')
        self.write("shape.obj")
        with self.assertRaisesRegex(ValueError, "Unrecognized resource"):
            runtime_assets(self.root, ["a.xml"])

    def test_pinned_registry_closure(self) -> None:
        """Verify the generated filegroup against every pinned registry task."""
        repo = Path(__file__).parents[2]
        source = os.environ.get("HUMANOID_BENCH_SOURCE")
        source_root = (
            Path(source)
            if source
            else Path(os.environ["TEST_SRCDIR"]) / "humanoid_bench_source"
        )
        self.assertEqual(
            (repo / "third_party/humanoid_bench/assets.BUILD").read_text(),
            build_file(
                source_root / "humanoid_bench/assets",
                repo / "envpool/mujoco/humanoid_bench/registry.json",
            ),
        )


if __name__ == "__main__":
    unittest.main()
