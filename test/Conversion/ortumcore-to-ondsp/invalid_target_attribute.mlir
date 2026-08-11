// RUN: not ondrix-opt %s --convert-ortumcore-to-ondsp-emulation 2>&1 | FileCheck %s

module attributes {
  test.rounding = #ortumcore<cx_rounding toward_negative>
} {
  // CHECK: 'builtin.module' op attribute 'test.rounding' contains an OrtumCore type or attribute
  func.func @metadata_only() {
    return
  }
}
