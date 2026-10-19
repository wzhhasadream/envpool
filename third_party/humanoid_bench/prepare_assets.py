"""Copy pinned HumanoidBench assets and apply the MuJoCo XML schema fix."""

import importlib.util
import importlib.metadata
import json
import shutil
import subprocess
import sys
from pathlib import Path
from xml.etree import ElementTree as ET

try:
    provenance = json.loads(importlib.metadata.distribution("humanoid-bench").read_text("direct_url.json") or "{}")
except importlib.metadata.PackageNotFoundError:
    provenance = {}
if provenance.get("vcs_info", {}).get("commit_id") != "d198f7b49cf889d86e41fdcba6b174e6b235fd12":
    subprocess.run([sys.executable, "-m", "pip", "install", "--no-deps", "humanoid-bench @ git+https://github.com/wzhhasadream/humanoid-bench-compatible.git@d198f7b49cf889d86e41fdcba6b174e6b235fd12"], check=True)
source = Path(next(iter(importlib.util.find_spec("humanoid_bench").submodule_search_locations)))
destination = Path(__file__).resolve().parents[2] / "envpool/mujoco/humanoid_bench/assets"
shutil.copytree(source / "assets", destination, dirs_exist_ok=True)
kettle = destination / "kitchen/adept_models/kitchen/assets/kettle_asset.xml"
tree = ET.parse(kettle)
root = tree.getroot()
default = root.find("default")
if default.get("class") == "kettle":
    root.remove(default)
    ET.SubElement(root, "default").append(default)
tree.write(kettle, encoding="unicode")
package = destination.parent
shutil.copy2(Path(__file__).with_name("LICENSE"), package / "LICENSE")
shutil.copy2(Path(__file__).with_name("NOTICE"), package / "NOTICE")
print(destination)
