// RUN: ondrix-opt %s --convert-ondrix-to-ondsp | FileCheck %s

// The wider profile is a different construction, not the Q15 one widened:
// a 1024-entry Q31 table (the pinned prefix decodes little-endian to the
// independently derived mpmath values 0, 13176712, 26352928, ...), the
// residual angle and its square and cube, three products combined into one
// correction, and then the SAME two boundaries the Q15 profile has -- one
// nearest-even round_shift and one saturating combine. Cosine differs only
// by the quarter-turn phase offset, 2^30 at this width.

// CHECK-LABEL: func.func @sine16_q31
// CHECK: arith.constant dense<"0x00000000880FC900201D9201D726
// CHECK: arith.constant 1727108826179 : i64
// CHECK: scf.for
// CHECK: arith.extui
// CHECK: arith.shrui %{{.*}}, %{{.*}} : i64
// CHECK: arith.divsi
// CHECK: ondsp.round_shift
// CHECK-SAME: post_shift_right = 31
// CHECK-SAME: rounding = nearest_even
// CHECK: ondsp.sat_cast
// CHECK-NOT: ondrix.sine
func.func @sine16_q31(%phase: tensor<16xi32>) -> tensor<16xi32> {
  %result = ondrix.sine %phase {
    numeric = #ondsp.fixed<signed, storage = i32, frac = 31>,
    rounding = #ondsp.rounding<nearest_even>
  } : (tensor<16xi32>) -> tensor<16xi32>
  return %result : tensor<16xi32>
}

// The quarter turn follows the phase width rather than staying at 16384.
// CHECK-LABEL: func.func @cosine16_q31
// CHECK: arith.constant 1073741824 : i64
// CHECK-NOT: ondrix.cosine
func.func @cosine16_q31(%phase: tensor<16xi32>) -> tensor<16xi32> {
  %result = ondrix.cosine %phase {
    numeric = #ondsp.fixed<signed, storage = i32, frac = 31>,
    rounding = #ondsp.rounding<nearest_even>
  } : (tensor<16xi32>) -> tensor<16xi32>
  return %result : tensor<16xi32>
}
