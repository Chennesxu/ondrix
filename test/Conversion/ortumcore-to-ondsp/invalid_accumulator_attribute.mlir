// RUN: not ondrix-opt %s --convert-ortumcore-to-ondsp-emulation 2>&1 | FileCheck %s

module attributes {
  test.acc_type = [{nested = !ortumcore.acc}]
} {
  // CHECK: 'builtin.module' op attribute 'test.acc_type' contains a target accumulator type
  func.func @metadata_only() {
    return
  }
}
