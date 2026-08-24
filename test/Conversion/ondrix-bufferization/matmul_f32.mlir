// RUN: ondrix-opt %s --one-shot-bufferize="bufferize-function-boundaries function-boundary-type-conversion=identity-layout-map allow-return-allocs" | FileCheck %s --implicit-check-not=ondsp.reduce_mac
// RUN: ondrix-opt %s --one-shot-bufferize="bufferize-function-boundaries function-boundary-type-conversion=identity-layout-map allow-return-allocs" --canonicalize --vectorize-ondsp-fp-filter-outputs="vector-width=4 supports-vector-fma=true" | FileCheck %s --check-prefix=BATCHED --implicit-check-not=ondsp.reduce_mac

// The f32 profile bufferizes straight to the column-tile loop form of its
// declared event graph. No transposed pack of B, hence the single allocation:
// the column axis is left for the output batching to read.

// CHECK-LABEL: func.func @f32_matmul
// CHECK: %[[OUT:.*]] = memref.alloc() {alignment = 64 : i64} : memref<4x6xf32>
// CHECK-NOT: memref.alloc
// CHECK: scf.for %[[ROW:.*]] = %c0
// CHECK: scf.for %[[COLUMN:.*]] = %c0{{.*}} to %c6
// CHECK: %[[ACC:.*]] = scf.for %[[TERM:.*]] = %c0{{.*}} to %c8{{.*}} step %c1{{.*}} iter_args(%[[SUM:.*]] = %cst) -> (f32)
// CHECK: %[[A:.*]] = memref.load %{{.*}}[%[[ROW]], %[[TERM]]] : memref<4x8xf32>
// CHECK: %[[B:.*]] = memref.load %{{.*}}[%[[TERM]], %[[COLUMN]]] : memref<8x6xf32>
// CHECK: math.fma %[[A]], %[[B]], %[[SUM]] {ondsp.fast_used = ["fuse_multiply_add"]} : f32
// CHECK: memref.store %[[ACC]], %[[OUT]][%[[ROW]], %[[COLUMN]]]

// BATCHED-LABEL: func.func @f32_matmul
// BATCHED: %[[BROW:.*]] = memref.load %{{.*}} : memref<4x8xf32>
// BATCHED: vector.splat %[[BROW]] : vector<4xf32>
// BATCHED: vector.load %{{.*}} : memref<8x6xf32>, vector<4xf32>
// BATCHED: math.fma {{.*}} {ondsp.fast_used = ["fuse_multiply_add"]} : vector<4xf32>
// BATCHED: vector.store {{.*}} : memref<4x6xf32>, vector<4xf32>
// 6 columns at width 4 leave two ordered columns behind the one full block.
// BATCHED: scf.for %{{.*}} = %c4
// BATCHED: math.fma {{.*}} : f32
func.func @f32_matmul(%a: tensor<4x8xf32>, %b: tensor<8x6xf32>) -> tensor<4x6xf32> {
  %c = ondrix.matmul %a, %b {
    numeric = #ondsp.fp<format = f32, contract = fast>
  } : (tensor<4x8xf32>, tensor<8x6xf32>) -> tensor<4x6xf32>
  return %c : tensor<4x6xf32>
}
