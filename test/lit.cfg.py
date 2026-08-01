# -*- Python -*-

import os
import shutil
import subprocess

import lit.formats
from lit.llvm import llvm_config

config.name = "ONDRIX"
config.test_format = lit.formats.ShTest(not llvm_config.use_lit_shell)
config.suffixes = [".mlir"]
config.test_source_root = os.path.dirname(__file__)
config.test_exec_root = os.path.join(config.ondrix_obj_root, "test")

llvm_config.with_system_environment(["HOME", "INCLUDE", "LIB", "TMP", "TEMP"])
llvm_config.use_default_substitutions()

config.excludes = ["Inputs", "CMakeLists.txt", "README.txt", "LICENSE.txt"]

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
