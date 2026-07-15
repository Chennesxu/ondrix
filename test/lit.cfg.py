# -*- Python -*-

import os

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

config.ondrix_tools_dir = os.path.join(config.ondrix_obj_root, "bin")
tool_dirs = [config.ondrix_tools_dir, config.llvm_tools_dir]
tools = [
    "ondrix-opt",
    "ondrix-translate",
    "ondrix-fixed-point-semantics-test",
    "FileCheck",
    "llc",
    "not",
]

llvm_config.add_tool_substitutions(tools, tool_dirs)
