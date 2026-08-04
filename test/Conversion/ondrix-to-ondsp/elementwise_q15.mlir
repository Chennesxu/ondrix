// RUN: ondrix-opt %s --convert-ondrix-to-ondsp --split-input-file | FileCheck %s

// Each member is one exact integer expression plus the boundary its contract
// names, and the boundary is the only ondsp operation in the body.

// The exact sum lives in add_shift's own carrier, so the loop body has no
// widening at all and the declared overflow rides the scale.
// CHECK-LABEL: func.func @add
// CHECK: ondsp.add_shift {{.*}}post_shift_right = 0{{.*}}overflow = wrap, saturate_to = i16
// CHECK-NOT: arith.extsi
func.func @add(%a: tensor<8xi16>, %b: tensor<8xi16>) -> tensor<8xi16> {
  %0 = ondrix.add %a, %b {
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    overflow = #ondsp.overflow<wrap>
  } : (tensor<8xi16>, tensor<8xi16>) -> tensor<8xi16>
  return %0 : tensor<8xi16>
}

// -----

// CHECK-LABEL: func.func @mult
// CHECK: arith.muli
// CHECK: ondsp.round_shift {{.*}}post_shift_right = 15, rounding = toward_zero, overflow = saturate, saturate_to = i16
func.func @mult(%a: tensor<8xi16>, %b: tensor<8xi16>) -> tensor<8xi16> {
  %0 = ondrix.mult %a, %b {
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    rounding = #ondsp.rounding<toward_zero>,
    overflow = #ondsp.overflow<saturate>
  } : (tensor<8xi16>, tensor<8xi16>) -> tensor<8xi16>
  return %0 : tensor<8xi16>
}

// -----

// The magnitude is taken at i32, which is what keeps |-32768| exact and
// leaves the declared overflow as the only thing that decides that input.
// CHECK-LABEL: func.func @abs
// CHECK: arith.extsi %{{.*}} : i16 to i32
// CHECK: arith.maxsi
// CHECK: ondsp.round_shift {{.*}}post_shift_right = 0{{.*}}overflow = saturate, saturate_to = i16
func.func @abs(%a: tensor<8xi16>) -> tensor<8xi16> {
  %0 = ondrix.abs %a {
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    overflow = #ondsp.overflow<saturate>
  } : (tensor<8xi16>) -> tensor<8xi16>
  return %0 : tensor<8xi16>
}

// -----

// A left shift cannot round, so the boundary shifts by zero and only the
// declared overflow acts; a right shift is the opposite.
// CHECK-LABEL: func.func @shift_left
// CHECK: arith.shli
// CHECK: ondsp.round_shift {{.*}}post_shift_right = 0{{.*}}overflow = wrap, saturate_to = i16
func.func @shift_left(%a: tensor<8xi16>) -> tensor<8xi16> {
  %0 = ondrix.shift %a {
    amount = 4 : i64,
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    rounding = #ondsp.rounding<nearest_even>,
    overflow = #ondsp.overflow<wrap>
  } : (tensor<8xi16>) -> tensor<8xi16>
  return %0 : tensor<8xi16>
}

// -----

// CHECK-LABEL: func.func @shift_right
// CHECK-NOT: arith.shli
// CHECK: ondsp.round_shift {{.*}}post_shift_right = 3, rounding = nearest_ties_positive
func.func @shift_right(%a: tensor<8xi16>) -> tensor<8xi16> {
  %0 = ondrix.shift %a {
    amount = -3 : i64,
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    rounding = #ondsp.rounding<nearest_ties_positive>,
    overflow = #ondsp.overflow<saturate>
  } : (tensor<8xi16>) -> tensor<8xi16>
  return %0 : tensor<8xi16>
}
