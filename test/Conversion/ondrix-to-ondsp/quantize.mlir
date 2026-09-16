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

// The domain change keeps its own boundary operation, carrying the policy.
// CHECK-LABEL: func.func @quantize_f32
// CHECK: scf.for
// CHECK: ondsp.convert {{.*}}rounding = #ondsp.rounding<nearest_even>{{.*}} : (f32) -> i16
func.func @quantize_f32(%x: tensor<8xf32>) -> tensor<8xi16> {
  %0 = ondrix.quantize %x {
    src = #ondsp.fp<format = f32, contract = off>,
    dst = #ondsp.fixed<signed, storage = i16, frac = 15>,
    rounding = #ondsp.rounding<nearest_even>,
    overflow = #ondsp.overflow<saturate>
  } : (tensor<8xf32>) -> tensor<8xi16>
  return %0 : tensor<8xi16>
}

// -----

// CHECK-LABEL: func.func @dequantize_q31
// CHECK: ondsp.convert
// CHECK-NOT: rounding
// CHECK-SAME: : (i32) -> f32
func.func @dequantize_q31(%x: tensor<8xi32>) -> tensor<8xf32> {
  %0 = ondrix.quantize %x {
    src = #ondsp.fixed<signed, storage = i32, frac = 31>,
    dst = #ondsp.fp<format = f32, contract = off>
  } : (tensor<8xi32>) -> tensor<8xf32>
  return %0 : tensor<8xf32>
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
