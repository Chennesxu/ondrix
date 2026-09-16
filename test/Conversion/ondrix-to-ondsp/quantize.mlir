// RUN: ondrix-opt %s --convert-ondrix-to-ondsp --split-input-file | FileCheck %s

// A widening has no boundary, so its body has no ondsp operation at all:
// the sign extension and the exact shift by the width difference.
// CHECK-LABEL: func.func @widen
// CHECK: scf.for
// CHECK: %[[WIDE:.*]] = arith.extsi %{{.*}} : i16 to i32
// CHECK: arith.shli %[[WIDE]], %c16_i32
// CHECK-NOT: ondsp.
func.func @widen(%x: tensor<8xi16>) -> tensor<8xi32> {
  %0 = ondrix.quantize %x {
    src = #ondsp.fixed<signed, storage = i16, frac = 15>,
    dst = #ondsp.fixed<signed, storage = i32, frac = 31>
  } : (tensor<8xi16>) -> tensor<8xi32>
  return %0 : tensor<8xi32>
}

// -----

// A narrowing is the one declared boundary and nothing before it: no
// widening carrier, the shift by the width difference, the declared policy.
// CHECK-LABEL: func.func @narrow
// CHECK-NOT: arith.extsi
// CHECK: ondsp.round_shift {{.*}}pre_shift_left = 0, post_shift_right = 16, rounding = nearest_even, overflow = wrap, saturate_to = i16
// CHECK-SAME: (i32) -> i16
func.func @narrow(%x: tensor<8xi32>) -> tensor<8xi16> {
  %0 = ondrix.quantize %x {
    src = #ondsp.fixed<signed, storage = i32, frac = 31>,
    dst = #ondsp.fixed<signed, storage = i16, frac = 15>,
    rounding = #ondsp.rounding<nearest_even>,
    overflow = #ondsp.overflow<wrap>
  } : (tensor<8xi32>) -> tensor<8xi16>
  return %0 : tensor<8xi16>
}

// -----

// The scalar form is the same body without the loop.
// CHECK-LABEL: func.func @scalar
// CHECK-NOT: scf.for
// CHECK: ondsp.round_shift %arg0 {{.*}}post_shift_right = 16, rounding = nearest_ties_positive, overflow = saturate, saturate_to = i16
func.func @scalar(%x: i32) -> i16 {
  %0 = ondrix.quantize %x {
    src = #ondsp.fixed<signed, storage = i32, frac = 31>,
    dst = #ondsp.fixed<signed, storage = i16, frac = 15>,
    rounding = #ondsp.rounding<nearest_ties_positive>,
    overflow = #ondsp.overflow<saturate>
  } : (i32) -> i16
  return %0 : i16
}
