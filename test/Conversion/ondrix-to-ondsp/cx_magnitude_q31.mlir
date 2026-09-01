// RUN: ondrix-opt %s --convert-ondrix-to-ondsp | FileCheck %s

// Two Q31 squares reach exactly 2^63, one past the signed i64 maximum, so each
// component carries a one-bit declared boundary and the root restores it. The
// restoration is applied to the SUM, before the root, so the low bit is
// resolved rather than zeroed.

// CHECK-LABEL: func.func @magnitude_q31
// CHECK: ondsp.round_shift
// CHECK-SAME: post_shift_right = 1
// CHECK-SAME: rounding = toward_negative
// CHECK-SAME: saturate_to = i32
// CHECK: arith.muli
// CHECK: %[[RESTORE:.*]] = arith.constant 2 : i64
// CHECK: arith.shli %{{.*}}, %[[RESTORE]]
// CHECK: ondsp.sqrt_fixed
// CHECK-SAME: (i64) -> i32
// CHECK-NOT: ondrix.cx_magnitude
func.func @magnitude_q31(%input: tensor<4xi64>) -> tensor<4xi32> {
  %result = ondrix.cx_magnitude %input {
    layout = #ondsp.cx_layout<packed_i32_imag_hi_real_lo>,
    numeric = #ondsp.fixed<signed, storage = i32, frac = 31>,
    input_rounding = #ondsp.rounding<toward_negative>,
    rounding = #ondsp.rounding<nearest_even>
  } : (tensor<4xi64>) -> tensor<4xi32>
  return %result : tensor<4xi32>
}

// The Q15 sum of two squares reaches only 2^31 and stays exact, so that
// profile emits no component boundary and no restoration at all.
// CHECK-LABEL: func.func @magnitude_q15
// CHECK-NOT: ondsp.round_shift
// CHECK-NOT: arith.shli
// CHECK: ondsp.sqrt_fixed
// CHECK-SAME: (i64) -> i16
func.func @magnitude_q15(%input: tensor<4xi32>) -> tensor<4xi16> {
  %result = ondrix.cx_magnitude %input {
    layout = #ondsp.cx_layout<packed_i16_imag_hi_real_lo>,
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    rounding = #ondsp.rounding<nearest_even>
  } : (tensor<4xi32>) -> tensor<4xi16>
  return %result : tensor<4xi16>
}
