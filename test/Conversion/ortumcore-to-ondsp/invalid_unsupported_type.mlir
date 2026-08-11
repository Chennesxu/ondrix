// RUN: not ondrix-opt %s --convert-ortumcore-to-ondsp-emulation 2>&1 | FileCheck %s

// CHECK: unsupported or nested OrtumCore type
func.func @unsupported_type(%accs: tensor<4x!ortumcore.acc>) -> tensor<4x!ortumcore.acc> {
  return %accs : tensor<4x!ortumcore.acc>
}
