// RUN: ondrix-opt %s | FileCheck %s

// The runtime-divisor division: the divisor is an operand and the policy
// for a divisor that is not positive is spelled on the operation.
// CHECK-LABEL: func.func @ratio_q15
func.func @ratio_q15(%x: i16, %y: i16) -> i16 {
  // CHECK: ondsp.round_quotient %arg0, %arg1
  // CHECK-SAME: nonpositive = #ondsp.nonpositive_divisor<trap>
  // CHECK-SAME: pre_shift_left = 15
  // CHECK-SAME: rounding = #ondsp.rounding<nearest_ties_positive>
  %0 = ondsp.round_quotient %x, %y {
    pre_shift_left = 15 : i64,
    rounding = #ondsp.rounding<nearest_ties_positive>,
    overflow = #ondsp.overflow<saturate>,
    nonpositive = #ondsp.nonpositive_divisor<trap>
  } : (i16, i16) -> i16
  return %0 : i16
}

// CHECK-LABEL: func.func @saturating_policy
func.func @saturating_policy(%x: i64, %y: i32) -> i32 {
  // CHECK: ondsp.round_quotient
  // CHECK-SAME: nonpositive = #ondsp.nonpositive_divisor<saturate>
  // CHECK-SAME: overflow = #ondsp.overflow<wrap>
  %0 = ondsp.round_quotient %x, %y {
    pre_shift_left = 0 : i64,
    rounding = #ondsp.rounding<toward_zero>,
    overflow = #ondsp.overflow<wrap>,
    nonpositive = #ondsp.nonpositive_divisor<saturate>
  } : (i64, i32) -> i32
  return %0 : i32
}
