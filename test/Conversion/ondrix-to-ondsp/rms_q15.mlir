// RUN: ondrix-opt %s --convert-ondrix-to-ondsp | FileCheck %s

// One exact square-accumulation loop, then exactly the two declared
// boundaries: the nearest-even mean by log2(N) into i32 (unreachable
// saturation) and the integer square root of the structurally
// nonnegative mean.

// CHECK-LABEL: func.func @rms16_q15
// CHECK: scf.for
// CHECK: arith.muli
// CHECK: ondsp.round_shift
// CHECK-SAME: post_shift_right = 4
// CHECK-SAME: saturate_to = i32
// CHECK: ondsp.sqrt_fixed
// CHECK-NOT: ondrix.rms
func.func @rms16_q15(%input: tensor<16xi16>) -> tensor<1xi16> {
  %result = ondrix.rms %input {
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    rounding = #ondsp.rounding<nearest_even>
  } : (tensor<16xi16>) -> tensor<1xi16>
  return %result : tensor<1xi16>
}
