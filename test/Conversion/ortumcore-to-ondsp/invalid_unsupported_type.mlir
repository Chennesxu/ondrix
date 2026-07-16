// RUN: not ondrix-opt %s --convert-ortumcore-to-ondsp-emulation 2>&1 | FileCheck %s

// CHECK: unsupported or nested OrtumCore type
func.func @unsupported_type(%state: !ortumcore.vec.state) -> !ortumcore.vec.state {
  return %state : !ortumcore.vec.state
}
