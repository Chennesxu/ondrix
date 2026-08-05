// RUN: ondrix-opt %s | ondrix-opt | FileCheck %s

// CHECK-LABEL: func.func @cx_phase_q15
// CHECK: ondrix.cx_phase
// CHECK-SAME: layout = #ondsp.cx_layout<packed_i16_imag_hi_real_lo>
// CHECK-SAME: output_numeric = #ondsp.fixed<unsigned, storage = i16, frac = 16>
// CHECK-SAME: rounding = #ondsp.rounding<nearest_even>
func.func @cx_phase_q15(%input: tensor<64xi32>) -> tensor<64xi16> {
  %result = ondrix.cx_phase %input {
    layout = #ondsp.cx_layout<packed_i16_imag_hi_real_lo>,
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    output_numeric = #ondsp.fixed<unsigned, storage = i16, frac = 16>,
    rounding = #ondsp.rounding<nearest_even>
  } : (tensor<64xi32>) -> tensor<64xi16>
  return %result : tensor<64xi16>
}

// The result is the same unsigned turn ondrix.sine reads, so composing the
// two is a well-formed program with no conversion between them.
// CHECK-LABEL: func.func @phase_feeds_sine
// CHECK: ondrix.cx_phase
// CHECK: ondrix.sine
func.func @phase_feeds_sine(%input: tensor<33xi32>) -> tensor<33xi16> {
  %turn = ondrix.cx_phase %input {
    layout = #ondsp.cx_layout<packed_i16_imag_hi_real_lo>,
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    output_numeric = #ondsp.fixed<unsigned, storage = i16, frac = 16>,
    rounding = #ondsp.rounding<nearest_even>
  } : (tensor<33xi32>) -> tensor<33xi16>
  %wave = ondrix.sine %turn {
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    rounding = #ondsp.rounding<nearest_even>
  } : (tensor<33xi16>) -> tensor<33xi16>
  return %wave : tensor<33xi16>
}
