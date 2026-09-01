// RUN: ondrix-opt %s --convert-ondrix-to-ondsp | FileCheck %s

// Only the tap sum can leave i64, so only it carries a per-product boundary.
// The step and the weight update are each a single product and keep the full
// shift; the output boundary absorbs what the product boundary took.

// CHECK-LABEL: func.func @lms_k32_q31
// CHECK: ondsp.round_shift
// CHECK-SAME: post_shift_right = 5
// CHECK-SAME: rounding = toward_negative
// CHECK-SAME: saturate_to = i64
// CHECK: ondsp.round_shift
// CHECK-SAME: post_shift_right = 26
// CHECK-SAME: saturate_to = i32
// CHECK: ondsp.round_shift
// CHECK-SAME: post_shift_right = 31
// CHECK-NOT: ondrix.lms
func.func @lms_k32_q31(%x: tensor<16xi32>, %d: tensor<16xi32>, %w: tensor<32xi32>)
    -> (tensor<16xi32>, tensor<32xi32>) {
  %e, %a = ondrix.lms %x, %d, %w {
    numeric = #ondsp.fixed<signed, storage = i32, frac = 31>,
    step_size = 268435456 : i64,
    product_rounding = #ondsp.rounding<toward_negative>,
    rounding = #ondsp.rounding<nearest_even>
  } : (tensor<16xi32>, tensor<16xi32>, tensor<32xi32>) -> (tensor<16xi32>, tensor<32xi32>)
  return %e, %a : tensor<16xi32>, tensor<32xi32>
}

// A different tap count derives a different shift, and K=5 is deliberately not
// a power of two: a ceil derivation would read 3/28 here and 5/26 above.
// CHECK-LABEL: func.func @lms_k5_q31
// CHECK: ondsp.round_shift
// CHECK-SAME: post_shift_right = 2
// CHECK: ondsp.round_shift
// CHECK-SAME: post_shift_right = 29
func.func @lms_k5_q31(%x: tensor<16xi32>, %d: tensor<16xi32>, %w: tensor<5xi32>)
    -> (tensor<16xi32>, tensor<5xi32>) {
  %e, %a = ondrix.lms %x, %d, %w {
    numeric = #ondsp.fixed<signed, storage = i32, frac = 31>,
    step_size = 1024 : i64,
    product_rounding = #ondsp.rounding<nearest_even>,
    rounding = #ondsp.rounding<nearest_even>
  } : (tensor<16xi32>, tensor<16xi32>, tensor<5xi32>) -> (tensor<16xi32>, tensor<5xi32>)
  return %e, %a : tensor<16xi32>, tensor<5xi32>
}
