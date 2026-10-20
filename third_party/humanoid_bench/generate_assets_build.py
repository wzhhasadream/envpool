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

"""Generate the pinned asset filegroup from every registered task's XML.

Pass an extracted humanoid-bench-compatible source tree. This tool reads XML
and copies no executable upstream code. Use --check after updating the registry
or upstream pin to verify that assets.BUILD still covers the complete closure.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from xml.etree import ElementTree as ET

_TEXTURE_FILES = {
    "file",
    "fileright",
    "fileleft",
    "fileup",
    "filedown",
    "filefront",
    "fileback",
}
_ASSET_TAGS = {"mesh", "skin", "flexcomp", "hfield", "texture"}


def _resolve(root: Path, model_dir: Path, xml_dir: Path, filename: str) -> Path:
    """Match MuJoCo 3.6's model-directory-first, include-directory fallback."""
    if Path(filename).is_absolute():
        raise ValueError(f"Asset path must be relocatable: {filename}")
    for directory in (model_dir, xml_dir):
        # Preserve Bazel's runfiles paths: each input file may be a symlink.
        path = Path(os.path.abspath(directory / filename))
        if not path.is_relative_to(root):
            raise ValueError(f"Asset path escapes the asset root: {filename}")
        if path.is_file():
            return path
    raise FileNotFoundError(
        f"Missing asset {filename!r} referenced in {xml_dir}"
    )


def _xml_elements(
    root: Path, model: Path
) -> tuple[list[tuple[ET.Element, Path]], set[Path]]:
    elements: list[tuple[ET.Element, Path]] = []
    visited: set[Path] = set()

    def read_xml(path: Path) -> None:
        if path in visited:
            return
        visited.add(path)
        for element in ET.parse(path).iter():
            if element.tag in {"include", "model"}:
                included = _resolve(
                    root, model.parent, path.parent, element.attrib["file"]
                )
                read_xml(included)
            else:
                elements.append((element, path.parent))

    read_xml(model)
    return elements, visited


def runtime_assets(asset_root: Path, task_xmls: list[str]) -> list[str]:
    """Find all reachable model resources and retain every asset license."""
    root = Path(os.path.abspath(asset_root))
    files: set[Path] = set()
    for task_xml in sorted(set(task_xmls)):
        model = _resolve(root, root, root, task_xml)
        elements, xml_files = _xml_elements(root, model)
        files.update(xml_files)
        compiler: dict[str, str] = {}
        for element, _ in elements:
            if element.tag == "compiler":
                compiler.update(element.attrib)
        for element, xml_dir in elements:
            for attribute, filename in element.attrib.items():
                if not attribute.startswith("file"):
                    continue
                if element.tag not in _ASSET_TAGS or (
                    attribute != "file"
                    and not (
                        element.tag == "texture" and attribute in _TEXTURE_FILES
                    )
                ):
                    raise ValueError(
                        f"Unrecognized resource {element.tag}/{attribute}"
                    )
                prefix = compiler.get("assetdir", "")
                if element.tag == "texture":
                    prefix = compiler.get("texturedir", prefix)
                elif element.tag in {"mesh", "skin", "flexcomp"}:
                    prefix = compiler.get("meshdir", prefix)
                if compiler.get("strippath", "false") == "true":
                    filename = Path(filename).name
                files.add(
                    _resolve(
                        root,
                        model.parent,
                        xml_dir,
                        str(Path(prefix) / filename),
                    )
                )
    for path in root.rglob("*"):
        if path.is_file() and path.stem.lower() in {
            "license",
            "copying",
            "notice",
        }:
            files.add(path)
    return sorted(path.relative_to(root).as_posix() for path in files)


def build_file(asset_root: Path, registry_path: Path) -> str:
    """Render a deterministic, declarative Bazel filegroup."""
    registry = json.loads(registry_path.read_text())
    files = runtime_assets(
        asset_root, [task["xml_file"] for task in registry["tasks"].values()]
    )
    header = Path(__file__).read_text().split('"""', 1)[0]
    lines = [
        header.rstrip(),
        "",
        "# Generated by generate_assets_build.py; do not edit by hand.",
        f"# Upstream revision: {registry['revision']}",
        f"# Includes all {len(registry['tasks'])} registered tasks and asset licenses.",
        "",
        "filegroup(",
        '    name = "assets",',
        "    srcs = [",
        *(f'        "humanoid_bench/assets/{path}",' for path in files),
        "    ],",
        '    visibility = ["//visibility:public"],',
        ")",
        "",
        "# Test input only: retain the full source to detect stale closure lists.",
        "filegroup(",
        '    name = "source_assets",',
        '    srcs = glob(["humanoid_bench/assets/**"]),',
        '    visibility = ["//visibility:public"],',
        ")",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    """Generate or verify assets.BUILD using the extracted, pinned source."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_root", type=Path)
    parser.add_argument(
        "--registry",
        type=Path,
        default=Path(__file__).resolve().parents[2]
        / "envpool/mujoco/humanoid_bench/registry.json",
    )
    parser.add_argument(
        "--output", type=Path, default=Path(__file__).with_name("assets.BUILD")
    )
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    content = build_file(
        args.source_root / "humanoid_bench/assets", args.registry
    )
    if args.check:
        if args.output.read_text() != content:
            raise SystemExit(f"Stale asset filegroup: regenerate {args.output}")
    else:
        args.output.write_text(content)


if __name__ == "__main__":
    main()
