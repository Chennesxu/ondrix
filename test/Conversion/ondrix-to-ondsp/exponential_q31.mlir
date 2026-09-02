// RUN: ondrix-opt %s --convert-ondrix-to-ondsp --split-input-file | FileCheck %s

// The Q31 profiles read 1024-point tables by a residual series rather than
// by interpolation; the residual ratio is one unsigned division.

// CHECK-LABEL: func.func @log2_q31
// CHECK: arith.constant dense<{{.*}}> : tensor<1024xi32>
// CHECK: math.ctlz
// CHECK: arith.divui
// One boundary: the return to Q6.26. The pole is a declared value.
// CHECK-COUNT-1: ondsp.round_shift {{.*}}post_shift_right = 30, rounding = nearest_even
// CHECK-NOT: ondsp.round_shift
// CHECK: arith.select
func.func @log2_q31(%a: tensor<8xi32>) -> tensor<8xi32> {
  %0 = ondrix.log2 %a {
    numeric = #ondsp.fixed<unsigned, storage = i32, frac = 32>,
    output_numeric = #ondsp.fixed<signed, storage = i32, frac = 26>,
    rounding = #ondsp.rounding<nearest_even>
  } : (tensor<8xi32>) -> tensor<8xi32>
  return %0 : tensor<8xi32>
}

// -----

// One boundary, the binade placement, written out because its amount is
// input-dependent; no round_shift and no interpolation remain.
// CHECK-LABEL: func.func @exp2_q31
// CHECK: arith.constant dense<{{.*}}> : tensor<1024xi32>
// CHECK: arith.shrui
// CHECK-NOT: ondsp.round_shift
// CHECK: arith.select
func.func @exp2_q31(%a: tensor<8xi32>) -> tensor<8xi32> {
  %0 = ondrix.exp2 %a {
    numeric = #ondsp.fixed<signed, storage = i32, frac = 26>,
    output_numeric = #ondsp.fixed<unsigned, storage = i32, frac = 32>,
    rounding = #ondsp.rounding<nearest_even>
  } : (tensor<8xi32>) -> tensor<8xi32>
  return %0 : tensor<8xi32>
}
