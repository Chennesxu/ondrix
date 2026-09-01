// RUN: ondrix-opt %s --convert-ondrix-to-ondsp | FileCheck %s

// The Q31 profile narrows each product before it joins the K-sum. Two inner
// extents derive two different shifts, so a lowering that pinned one fails the
// other; K = 1 derives none at all.

// CHECK-LABEL: func.func @matmul_k64_q31
// CHECK: arith.muli
// CHECK: ondsp.round_shift
// CHECK-SAME: post_shift_right = 6
// CHECK-SAME: rounding = toward_negative
// CHECK-SAME: saturate_to = i64
// CHECK: ondsp.round_shift
// CHECK-SAME: post_shift_right = 25
// CHECK-SAME: rounding = nearest_even
// CHECK-SAME: saturate_to = i32
// CHECK-NOT: ondrix.matmul
func.func @matmul_k64_q31(%a: tensor<2x64xi32>, %b: tensor<64x2xi32>) -> tensor<2x2xi32> {
  %result = ondrix.matmul %a, %b {
    numeric = #ondsp.fixed<signed, storage = i32, frac = 31>,
    product_rounding = #ondsp.rounding<toward_negative>,
    rounding = #ondsp.rounding<nearest_even>
  } : (tensor<2x64xi32>, tensor<64x2xi32>) -> tensor<2x2xi32>
  return %result : tensor<2x2xi32>
}

// CHECK-LABEL: func.func @matmul_k5_q31
// CHECK: ondsp.round_shift
// CHECK-SAME: post_shift_right = 2
// CHECK: ondsp.round_shift
// CHECK-SAME: post_shift_right = 29
func.func @matmul_k5_q31(%a: tensor<2x5xi32>, %b: tensor<5x2xi32>) -> tensor<2x2xi32> {
  %result = ondrix.matmul %a, %b {
    numeric = #ondsp.fixed<signed, storage = i32, frac = 31>,
    product_rounding = #ondsp.rounding<nearest_even>,
    rounding = #ondsp.rounding<nearest_even>
  } : (tensor<2x5xi32>, tensor<5x2xi32>) -> tensor<2x2xi32>
  return %result : tensor<2x2xi32>
}

// One product already fits i64, so this shape declares no product boundary and
// its export keeps the full 31.
// CHECK-LABEL: func.func @matmul_k1_q31
// CHECK: ondsp.round_shift
// CHECK-SAME: post_shift_right = 31
// CHECK-NOT: ondsp.round_shift
func.func @matmul_k1_q31(%a: tensor<2x1xi32>, %b: tensor<1x2xi32>) -> tensor<2x2xi32> {
  %result = ondrix.matmul %a, %b {
    numeric = #ondsp.fixed<signed, storage = i32, frac = 31>,
    rounding = #ondsp.rounding<nearest_even>
  } : (tensor<2x1xi32>, tensor<1x2xi32>) -> tensor<2x2xi32>
  return %result : tensor<2x2xi32>
}
