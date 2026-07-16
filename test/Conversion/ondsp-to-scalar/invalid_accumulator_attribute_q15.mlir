// RUN: not ondrix-opt %s --convert-ondsp-fixed-to-scalar 2>&1 | FileCheck %s

module attributes {
  test.acc_type = [{nested = !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>}]
} {
  // CHECK: 'builtin.module' op attribute 'test.acc_type' contains an Ondsp type or attribute
  func.func @metadata_only() {
    return
  }
}
