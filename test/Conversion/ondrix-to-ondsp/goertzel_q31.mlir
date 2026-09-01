// RUN: ondrix-opt %s --convert-ondrix-to-ondsp | FileCheck %s

// Two things the Q31 profile does that the Q15 one cannot: the step product
// folds the doubling into the shift, because 2*c*s1 reaches 2^63 at this
// width; and each of the three state values takes one declared bit before the
// squares, because three terms of 2^62 leave i64.

// CHECK-LABEL: func.func @goertzel_q31
// CHECK: ondsp.round_shift {{.*}}post_shift_right = 30, {{.*}}saturate_to = i32
// CHECK: ondsp.round_shift {{.*}}post_shift_right = 1, {{.*}}saturate_to = i64
// CHECK: ondsp.round_shift {{.*}}post_shift_right = 1, {{.*}}saturate_to = i64
// CHECK: ondsp.round_shift {{.*}}post_shift_right = 1, {{.*}}saturate_to = i64
// CHECK-NOT: ondrix.goertzel
func.func @goertzel_q31(%input: tensor<64xi32>) -> tensor<1xi64> {
  %energy = ondrix.goertzel %input {
    bin = 5 : i64,
    numeric = #ondsp.fixed<signed, storage = i32, frac = 31>,
    state_rounding = #ondsp.rounding<nearest_even>,
    rounding = #ondsp.rounding<nearest_even>
  } : (tensor<64xi32>) -> tensor<1xi64>
  return %energy : tensor<1xi64>
}

// The Q15 profile keeps its doubled spelling and its exact energy, so nothing
// here moved: shift 15 into i16 and no state boundary at all.
// CHECK-LABEL: func.func @goertzel_q15
// CHECK: ondsp.round_shift {{.*}}post_shift_right = 15, {{.*}}saturate_to = i16
// CHECK-NOT: saturate_to = i64
func.func @goertzel_q15(%input: tensor<64xi16>) -> tensor<1xi64> {
  %energy = ondrix.goertzel %input {
    bin = 5 : i64,
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    rounding = #ondsp.rounding<nearest_even>
  } : (tensor<64xi16>) -> tensor<1xi64>
  return %energy : tensor<1xi64>
}

// The neighbouring bin at the SAME extent is admissible, which is what makes
// the refusal in goertzel_q31_tie_refusal.mlir a per-bin decision.
// CHECK-LABEL: func.func @goertzel_q31_admissible_neighbour
// CHECK-NOT: ondrix.goertzel
func.func @goertzel_q31_admissible_neighbour(%input: tensor<2093xi32>) -> tensor<1xi64> {
  %energy = ondrix.goertzel %input {
    bin = 613 : i64,
    numeric = #ondsp.fixed<signed, storage = i32, frac = 31>,
    state_rounding = #ondsp.rounding<nearest_even>,
    rounding = #ondsp.rounding<nearest_even>
  } : (tensor<2093xi32>) -> tensor<1xi64>
  return %energy : tensor<1xi64>
}
