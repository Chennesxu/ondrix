// RUN: not ondrix-opt %s --convert-ondsp-q15-to-scalar 2>&1 | FileCheck %s

module attributes {
  test.acc_type = [{nested = !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>}]
} {
  // CHECK: failed to legalize operation 'builtin.module'
  func.func @metadata_only() {
    return
  }
}
