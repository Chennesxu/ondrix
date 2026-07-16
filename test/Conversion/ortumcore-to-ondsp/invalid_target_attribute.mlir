// RUN: not ondrix-opt %s --convert-ortumcore-to-ondsp-emulation 2>&1 | FileCheck %s

module attributes {
  test.stage = #ortumcore<fft_stage_kind radix2>
} {
  // CHECK: 'builtin.module' op attribute 'test.stage' contains an OrtumCore type or attribute
  func.func @metadata_only() {
    return
  }
}
