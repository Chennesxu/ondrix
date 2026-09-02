// RUN: ondrix-opt %s --convert-ondrix-to-ondsp --split-input-file | FileCheck %s

// The Q0.32 turn: two unsigned divisions (the ratio and the angle-addition
// residual), a 1025-entry table ending on the exact eighth turn (2^30 =
// 0x40000000, little-endian trailing bytes), and one round_shift of 33.

// CHECK-LABEL: func.func @phase_q15_turn32
// CHECK: arith.constant dense<"0x00000000{{.*}}00000040"> : tensor<1025xi32>
// CHECK-COUNT-2: arith.divui
// CHECK-COUNT-1: ondsp.round_shift {{.*}}post_shift_right = 33, rounding = nearest_even
// CHECK-NOT: ondsp.round_shift
// CHECK: arith.select
func.func @phase_q15_turn32(%input: tensor<8xi32>) -> tensor<8xi32> {
  %result = ondrix.cx_phase %input {
    layout = #ondsp.cx_layout<packed_i16_imag_hi_real_lo>,
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    output_numeric = #ondsp.fixed<unsigned, storage = i32, frac = 32>,
    rounding = #ondsp.rounding<nearest_even>
  } : (tensor<8xi32>) -> tensor<8xi32>
  return %result : tensor<8xi32>
}

// -----

// The same turn program over Q31 components: only the unpack width differs.
// CHECK-LABEL: func.func @phase_q31_turn32
// CHECK: arith.trunci {{.*}} : i64 to i32
// CHECK-COUNT-2: arith.divui
// CHECK-COUNT-1: ondsp.round_shift {{.*}}post_shift_right = 33, rounding = nearest_even
func.func @phase_q31_turn32(%input: tensor<8xi64>) -> tensor<8xi32> {
  %result = ondrix.cx_phase %input {
    layout = #ondsp.cx_layout<packed_i32_imag_hi_real_lo>,
    numeric = #ondsp.fixed<signed, storage = i32, frac = 31>,
    output_numeric = #ondsp.fixed<unsigned, storage = i32, frac = 32>,
    rounding = #ondsp.rounding<nearest_even>
  } : (tensor<8xi64>) -> tensor<8xi32>
  return %result : tensor<8xi32>
}
