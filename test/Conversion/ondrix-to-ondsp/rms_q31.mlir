// RUN: ondrix-opt %s --convert-ondrix-to-ondsp | FileCheck %s

// The Q31 profile emits three declared boundaries and one exact restoration.
// The pre-shift lands INSIDE the accumulation loop, one per sample, and the
// restoration lands BEFORE the root so the low bits are resolved rather than
// zeroed.

// CHECK-LABEL: func.func @rms64_q31
// CHECK: scf.for
// CHECK: ondsp.round_shift
// CHECK-SAME: post_shift_right = 3
// CHECK-SAME: rounding = nearest_even
// CHECK-SAME: saturate_to = i32
// CHECK: arith.muli
// CHECK: ondsp.round_shift
// CHECK-SAME: post_shift_right = 6
// CHECK-SAME: saturate_to = i64
// CHECK: %[[RESTORE:.*]] = arith.constant 6 : i64
// CHECK: arith.shli %{{.*}}, %[[RESTORE]]
// CHECK: ondsp.sqrt_fixed
// CHECK-NOT: ondrix.rms
func.func @rms64_q31(%input: tensor<64xi32>) -> tensor<1xi32> {
  %result = ondrix.rms %input {
    numeric = #ondsp.fixed<signed, storage = i32, frac = 31>,
    input_rounding = #ondsp.rounding<nearest_even>,
    rounding = #ondsp.rounding<nearest_even>
  } : (tensor<64xi32>) -> tensor<1xi32>
  return %result : tensor<1xi32>
}

// A different extent derives a different pre-shift, so a lowering that pinned
// one constant fails here.
// CHECK-LABEL: func.func @rms16_q31
// CHECK: ondsp.round_shift
// CHECK-SAME: post_shift_right = 2
// CHECK: %[[R16:.*]] = arith.constant 4 : i64
// CHECK: arith.shli %{{.*}}, %[[R16]]
func.func @rms16_q31(%input: tensor<16xi32>) -> tensor<1xi32> {
  %result = ondrix.rms %input {
    numeric = #ondsp.fixed<signed, storage = i32, frac = 31>,
    input_rounding = #ondsp.rounding<toward_negative>,
    rounding = #ondsp.rounding<nearest_even>
  } : (tensor<16xi32>) -> tensor<1xi32>
  return %result : tensor<1xi32>
}
