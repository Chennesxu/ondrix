// RUN: ondrix-opt %s | ondrix-opt | FileCheck %s

// CHECK-LABEL: func.func @matmul_q15
// CHECK: ondrix.matmul
// CHECK-SAME: frac = 15
// CHECK-SAME: rounding = #ondsp.rounding<nearest_even>
func.func @matmul_q15(%a: tensor<4x8xi16>, %b: tensor<8x3xi16>) -> tensor<4x3xi16> {
  %c = ondrix.matmul %a, %b {
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    rounding = #ondsp.rounding<nearest_even>
  } : (tensor<4x8xi16>, tensor<8x3xi16>) -> tensor<4x3xi16>
  return %c : tensor<4x3xi16>
}

// CHECK-LABEL: func.func @matmul_vector_shapes
// CHECK: ondrix.matmul
func.func @matmul_vector_shapes(%a: tensor<1x64xi16>, %b: tensor<64x1xi16>) -> tensor<1x1xi16> {
  %c = ondrix.matmul %a, %b {
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    rounding = #ondsp.rounding<nearest_even>
  } : (tensor<1x64xi16>, tensor<64x1xi16>) -> tensor<1x1xi16>
  return %c : tensor<1x1xi16>
}

// The f32 profile carries a contract and no rounding attribute: it has no
// requantization boundary to place.
// CHECK-LABEL: func.func @f32_matmul
// CHECK: ondrix.matmul
// CHECK-SAME: numeric = #ondsp.fp<format = f32, contract = fma>
// CHECK-NOT: rounding
func.func @f32_matmul(%a: tensor<3x4xf32>, %b: tensor<4x2xf32>) -> tensor<3x2xf32> {
  %c = ondrix.matmul %a, %b {
    numeric = #ondsp.fp<format = f32, contract = fma>
  } : (tensor<3x4xf32>, tensor<4x2xf32>) -> tensor<3x2xf32>
  return %c : tensor<3x2xf32>
}

// The Q31 profile declares a product boundary because the K-sum cannot be
// exact; the extent derives its shift, only the rounding is declared.
// CHECK-LABEL: func.func @matmul_k64_q31
// CHECK: ondrix.matmul
// CHECK-SAME: numeric = #ondsp.fixed<signed, storage = i32, frac = 31>
// CHECK-SAME: product_rounding = #ondsp.rounding<toward_negative>
// CHECK-SAME: rounding = #ondsp.rounding<nearest_ties_positive>
func.func @matmul_k64_q31(%a: tensor<4x64xi32>, %b: tensor<64x3xi32>) -> tensor<4x3xi32> {
  %result = ondrix.matmul %a, %b {
    numeric = #ondsp.fixed<signed, storage = i32, frac = 31>,
    product_rounding = #ondsp.rounding<toward_negative>,
    rounding = #ondsp.rounding<nearest_ties_positive>
  } : (tensor<4x64xi32>, tensor<64x3xi32>) -> tensor<4x3xi32>
  return %result : tensor<4x3xi32>
}

// One Q1.31 product already fits i64, so this shape has no product boundary
// and declaring one is refused (see invalid_matmul.mlir).
// CHECK-LABEL: func.func @matmul_k1_q31
// CHECK: ondrix.matmul
// CHECK-NOT: product_rounding
func.func @matmul_k1_q31(%a: tensor<4x1xi32>, %b: tensor<1x3xi32>) -> tensor<4x3xi32> {
  %result = ondrix.matmul %a, %b {
    numeric = #ondsp.fixed<signed, storage = i32, frac = 31>,
    rounding = #ondsp.rounding<nearest_even>
  } : (tensor<4x1xi32>, tensor<1x3xi32>) -> tensor<4x3xi32>
  return %result : tensor<4x3xi32>
}
