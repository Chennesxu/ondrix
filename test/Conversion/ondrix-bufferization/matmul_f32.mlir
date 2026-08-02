// RUN: ondrix-opt %s --one-shot-bufferize="bufferize-function-boundaries function-boundary-type-conversion=identity-layout-map allow-return-allocs" | FileCheck %s
// RUN: ondrix-opt %s --one-shot-bufferize="bufferize-function-boundaries function-boundary-type-conversion=identity-layout-map allow-return-allocs" --canonicalize --vectorize-ondsp-fp-fast-memref-reduce="vector-width=4" | FileCheck %s --check-prefix=FAST

// The f32 bufferization reuses the transposed pack of the fixed profile, so
// both reduction operands are unit-stride views and the schedule stage can
// serve them. Nothing follows the reduction: an f32 element has no
// requantization boundary.
// CHECK-LABEL: func.func @f32_matmul
// CHECK: ondsp.reduce_mac
// CHECK-SAME: numeric = #ondsp.fp<format = f32, contract = fast>
// CHECK-NOT: ondsp.acc_export
// FAST: vector.reduction
func.func @f32_matmul(%a: tensor<4x8xf32>, %b: tensor<8x4xf32>) -> tensor<4x4xf32> {
  %c = ondrix.matmul %a, %b {
    numeric = #ondsp.fp<format = f32, contract = fast>
  } : (tensor<4x8xf32>, tensor<8x4xf32>) -> tensor<4x4xf32>
  return %c : tensor<4x4xf32>
}
