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
