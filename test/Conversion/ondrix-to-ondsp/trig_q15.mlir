// RUN: ondrix-opt %s --convert-ondrix-to-ondsp | FileCheck %s

// Both operations share one table-plus-interpolation lowering: a
// 256-entry tie-guarded sine table constant (printed as a little-endian
// hex blob at this size; the pinned prefix decodes to the independently
// derived mpmath values 0, 804, 1608, 2411, 3212, ...), one loop, one Q8
// nearest-even interpolation round_shift, and one saturating combine per
// element. Cosine differs only by the exact quarter-turn phase offset
// constant 16384.

// CHECK-LABEL: func.func @sine16
// CHECK: arith.constant dense<"0x0000240348066B098C0CAB0FC812
// CHECK: arith.constant 0 : i32
// CHECK: scf.for
// CHECK: arith.extui
// CHECK: ondsp.round_shift
// CHECK-SAME: post_shift_right = 8
// CHECK: ondsp.sat_cast
// CHECK-NOT: ondrix.sine
func.func @sine16(%phase: tensor<16xi16>) -> tensor<16xi16> {
  %result = ondrix.sine %phase {
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    rounding = #ondsp.rounding<nearest_even>
  } : (tensor<16xi16>) -> tensor<16xi16>
  return %result : tensor<16xi16>
}

// CHECK-LABEL: func.func @cosine16
// CHECK: arith.constant 16384 : i32
// CHECK: scf.for
// CHECK: ondsp.round_shift
// CHECK: ondsp.sat_cast
// CHECK-NOT: ondrix.cosine
func.func @cosine16(%phase: tensor<16xi16>) -> tensor<16xi16> {
  %result = ondrix.cosine %phase {
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    rounding = #ondsp.rounding<nearest_even>
  } : (tensor<16xi16>) -> tensor<16xi16>
  return %result : tensor<16xi16>
}
