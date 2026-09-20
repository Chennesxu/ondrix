// RUN: ondrix-opt %s --convert-ondrix-to-ondsp | FileCheck %s

// The declared reading becomes the scale: a Q15 result shifts the exact sum
// right by 15, and reading it at the sum's own fraction shifts nothing.
func.func @power_q15(%input: tensor<8xi32>) -> tensor<8xi16> {
  %out = ondrix.cx_power %input {
    layout = #ondsp.cx_layout<packed_i16_imag_hi_real_lo>,
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    output_numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    rounding = #ondsp.rounding<nearest_ties_positive>
  } : (tensor<8xi32>) -> tensor<8xi16>
  return %out : tensor<8xi16>
}

// CHECK-LABEL: func.func @power_q15
// CHECK: scf.for
// CHECK: tensor.extract
// CHECK: ondsp.cx_power
// CHECK-SAME: post_shift_right = 15
// CHECK-SAME: rounding = nearest_ties_positive
// CHECK-SAME: saturate_to = i16
// CHECK: tensor.insert

func.func @power_full_range(%input: tensor<8xi32>) -> tensor<8xi32> {
  %out = ondrix.cx_power %input {
    layout = #ondsp.cx_layout<packed_i16_imag_hi_real_lo>,
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    output_numeric = #ondsp.fixed<signed, storage = i32, frac = 30>
  } : (tensor<8xi32>) -> tensor<8xi32>
  return %out : tensor<8xi32>
}

// CHECK-LABEL: func.func @power_full_range
// CHECK: ondsp.cx_power
// CHECK-SAME: post_shift_right = 0
// CHECK-SAME: saturate_to = i32
