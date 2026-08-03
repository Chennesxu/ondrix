# -*- Python -*-

import os
import shutil
import subprocess

import lit.formats
from lit.llvm import llvm_config

config.name = "ONDRIX"
config.test_format = lit.formats.ShTest(not llvm_config.use_lit_shell)
config.suffixes = [".mlir", ".ll"]
config.test_source_root = os.path.dirname(__file__)
config.test_exec_root = os.path.join(config.ondrix_obj_root, "test")

llvm_config.with_system_environment(["HOME", "INCLUDE", "LIB", "TMP", "TEMP"])
llvm_config.use_default_substitutions()

config.excludes = ["Inputs", "CMakeLists.txt", "README.txt", "LICENSE.txt"]

# Backend characterization tests name the target they measure, and a build
# configured for fewer targets must skip them rather than fail. LLVM's own
# suite spells this `<arch>-registered-target`; out of tree the list has to be
# read back from the tool.
def _add_registered_targets(llc):
    probe = subprocess.run([llc, "--version"], capture_output=True, text=True)
    if probe.returncode != 0:
        return
    lines = probe.stdout.splitlines()
    for index, line in enumerate(lines):
        if "Registered Targets" not in line:
            continue
        for entry in lines[index + 1 :]:
            name = entry.strip().split(" ", 1)[0]
            if not name:
                break
            config.available_features.add(name.lower() + "-registered-target")
        break

# The frozen-table regeneration check needs 50-digit mpmath; environments
# without it skip that test rather than fail it.
if shutil.which("python3"):
    probe = subprocess.run(["python3", "-c", "import mpmath"], capture_output=True)
    if probe.returncode == 0:
        config.available_features.add("mpmath")

config.ondrix_tools_dir = os.path.join(config.ondrix_obj_root, "bin")
tool_dirs = [config.ondrix_tools_dir, config.llvm_tools_dir]
tools = [
    "ondrix-opt",
    "ondrix-translate",
    "ondrix-compile",
    "ondrix-canonical-twiddle-analysis-test",
    "ondrix-constant-sequence-analysis-test",
    "ondrix-fixed-point-prefix-range-analysis-test",
    "ondrix-fixed-point-semantics-test",
    "ondrix-ondsp-semantics-test",
    "ondrix-ortumcore-target-profile-test",
    "FileCheck",
    "llc",
    "not",
]

llvm_config.add_tool_substitutions(tools, tool_dirs)

_add_registered_targets(os.path.join(config.llvm_tools_dir, "llc"))
