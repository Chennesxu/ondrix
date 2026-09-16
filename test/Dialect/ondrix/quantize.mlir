// RUN: ondrix-opt %s | ondrix-opt | FileCheck %s

// The attributes follow the boundary: a widening carries none, a narrowing
// carries the tie rule and the overflow policy.
// CHECK-LABEL: func.func @widen
// CHECK: ondrix.quantize
// CHECK-NOT: rounding
// CHECK-SAME: (tensor<8xi16>) -> tensor<8xi32>
func.func @widen(%x: tensor<8xi16>) -> tensor<8xi32> {
  %0 = ondrix.quantize %x {
    src = #ondsp.fixed<signed, storage = i16, frac = 15>,
    dst = #ondsp.fixed<signed, storage = i32, frac = 31>
  } : (tensor<8xi16>) -> tensor<8xi32>
  return %0 : tensor<8xi32>
}

// CHECK-LABEL: func.func @narrow
// CHECK: ondrix.quantize
// CHECK-SAME: overflow = #ondsp.overflow<wrap>
// CHECK-SAME: rounding = #ondsp.rounding<toward_zero>
// CHECK-SAME: (tensor<4096xi32>) -> tensor<4096xi16>
func.func @narrow(%x: tensor<4096xi32>) -> tensor<4096xi16> {
  %0 = ondrix.quantize %x {
    src = #ondsp.fixed<signed, storage = i32, frac = 31>,
    dst = #ondsp.fixed<signed, storage = i16, frac = 15>,
    rounding = #ondsp.rounding<toward_zero>,
    overflow = #ondsp.overflow<wrap>
  } : (tensor<4096xi32>) -> tensor<4096xi16>
  return %0 : tensor<4096xi16>
}

// CHECK-LABEL: func.func @quantize_f32
// CHECK: ondrix.quantize
// CHECK-SAME: dst = #ondsp.fixed<signed, storage = i32, frac = 31>
// CHECK-SAME: src = #ondsp.fp<format = f32, contract = off>
// CHECK-SAME: (tensor<8xf32>) -> tensor<8xi32>
func.func @quantize_f32(%x: tensor<8xf32>) -> tensor<8xi32> {
  %0 = ondrix.quantize %x {
    src = #ondsp.fp<format = f32, contract = off>,
    dst = #ondsp.fixed<signed, storage = i32, frac = 31>,
    rounding = #ondsp.rounding<nearest_ties_positive>,
    overflow = #ondsp.overflow<saturate>
  } : (tensor<8xf32>) -> tensor<8xi32>
  return %0 : tensor<8xi32>
}

// CHECK-LABEL: func.func @dequantize_q15
// CHECK: ondrix.quantize
// CHECK-NOT: rounding
// CHECK-SAME: (tensor<8xi16>) -> tensor<8xf32>
func.func @dequantize_q15(%x: tensor<8xi16>) -> tensor<8xf32> {
  %0 = ondrix.quantize %x {
    src = #ondsp.fixed<signed, storage = i16, frac = 15>,
    dst = #ondsp.fp<format = f32, contract = off>
  } : (tensor<8xi16>) -> tensor<8xf32>
  return %0 : tensor<8xf32>
}

// CHECK-LABEL: func.func @scalar
// CHECK: ondrix.quantize {{.*}} : (i32) -> i16
func.func @scalar(%x: i32) -> i16 {
  %0 = ondrix.quantize %x {
    src = #ondsp.fixed<signed, storage = i32, frac = 31>,
    dst = #ondsp.fixed<signed, storage = i16, frac = 15>,
    rounding = #ondsp.rounding<nearest_ties_positive>,
    overflow = #ondsp.overflow<saturate>
  } : (i32) -> i16
  return %0 : i16
}
