// RUN: ondrix-opt %s | FileCheck %s

// CHECK-LABEL: func.func @round_div_scalar
func.func @round_div_scalar(%sum: i64) -> i16 {
  // CHECK: ondsp.round_div
  // CHECK-SAME: divisor = 3
  // CHECK-SAME: overflow = #ondsp.overflow<saturate>
  // CHECK-SAME: pre_shift_left = 0
  // CHECK-SAME: rounding = #ondsp.rounding<nearest_even>
  %mean = ondsp.round_div %sum {
    divisor = 3 : i64, pre_shift_left = 0 : i64,
    rounding = #ondsp.rounding<nearest_even>,
    overflow = #ondsp.overflow<saturate>
  } : (i64) -> i16
  return %mean : i16
}

// The declared pre-scale widens the exact carrier; a power-of-two divisor is
// admitted and agrees with round_shift, which the execution gate checks.
// CHECK-LABEL: func.func @round_div_scaled_vector
func.func @round_div_scaled_vector(%values: vector<4xi32>) -> vector<4xi32> {
  // CHECK: ondsp.round_div
  // CHECK-SAME: divisor = 4
  // CHECK-SAME: pre_shift_left = 8
  %scaled = ondsp.round_div %values {
    divisor = 4 : i64, pre_shift_left = 8 : i64,
    rounding = #ondsp.rounding<toward_negative>,
    overflow = #ondsp.overflow<wrap>
  } : (vector<4xi32>) -> vector<4xi32>
  return %scaled : vector<4xi32>
}
