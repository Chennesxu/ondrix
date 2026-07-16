// RUN: not ondrix-opt %s --convert-ondsp-fixed-to-scalar 2>&1 | FileCheck %s

module attributes {
  test.numeric = #ondsp.fixed<signed, storage = i16, frac = 15>
} {
  // CHECK: 'builtin.module' op attribute 'test.numeric' contains an Ondsp type or attribute
  func.func @metadata_only() {
    return
  }
}
