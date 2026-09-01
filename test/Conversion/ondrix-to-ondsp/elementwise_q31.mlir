// RUN: ondrix-opt %s --convert-ondrix-to-ondsp --split-input-file | FileCheck %s

// The same members as the Q15 file at the wider width. What changes is the
// carrier the exact body runs in, and it is the only thing the width can
// change: the boundary attributes are the same three in the same order.

// The wider width takes the same shape with the carrier one width up, which
// is what the shift of 31 needs to be exact before the boundary.
// CHECK-LABEL: func.func @mult_q31
// CHECK: arith.extsi {{.*}} : i32 to i64
// CHECK: arith.muli {{.*}} : i64
// CHECK: ondsp.round_shift {{.*}}post_shift_right = 31, rounding = toward_zero, overflow = saturate, saturate_to = i32
func.func @mult_q31(%a: tensor<8xi32>, %b: tensor<8xi32>) -> tensor<8xi32> {
  %0 = ondrix.mult %a, %b {
    numeric = #ondsp.fixed<signed, storage = i32, frac = 31>,
    rounding = #ondsp.rounding<toward_zero>,
    overflow = #ondsp.overflow<saturate>
  } : (tensor<8xi32>, tensor<8xi32>) -> tensor<8xi32>
  return %0 : tensor<8xi32>
}

// -----

// The Q31 sum still needs no widening: add_shift carries it.
// CHECK-LABEL: func.func @add_q31
// CHECK: ondsp.add_shift {{.*}}post_shift_right = 0{{.*}}overflow = wrap, saturate_to = i32
// CHECK-NOT: arith.extsi
func.func @add_q31(%a: tensor<8xi32>, %b: tensor<8xi32>) -> tensor<8xi32> {
  %0 = ondrix.add %a, %b {
    numeric = #ondsp.fixed<signed, storage = i32, frac = 31>,
    overflow = #ondsp.overflow<wrap>
  } : (tensor<8xi32>, tensor<8xi32>) -> tensor<8xi32>
  return %0 : tensor<8xi32>
}

// -----

// Negating one width up is the whole reason INT32_MIN's magnitude survives
// to reach the declared boundary.
// CHECK-LABEL: func.func @abs_q31
// CHECK: arith.extsi {{.*}} : i32 to i64
// CHECK: arith.maxsi {{.*}} : i64
// CHECK: ondsp.round_shift {{.*}}post_shift_right = 0{{.*}}saturate_to = i32
func.func @abs_q31(%a: tensor<8xi32>) -> tensor<8xi32> {
  %0 = ondrix.abs %a {
    numeric = #ondsp.fixed<signed, storage = i32, frac = 31>,
    overflow = #ondsp.overflow<saturate>
  } : (tensor<8xi32>) -> tensor<8xi32>
  return %0 : tensor<8xi32>
}
