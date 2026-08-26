// RUN: ondrix-opt %s --one-shot-bufferize="bufferize-function-boundaries function-boundary-type-conversion=identity-layout-map allow-return-allocs" | FileCheck %s --implicit-check-not=ondsp.reduce_mac
// RUN: ondrix-opt %s --one-shot-bufferize="bufferize-function-boundaries function-boundary-type-conversion=identity-layout-map allow-return-allocs" --canonicalize --vectorize-ondsp-fp-filter-outputs="vector-width=4 supports-vector-fma=true" | FileCheck %s --check-prefix=BATCHED --implicit-check-not=ondsp.reduce_mac
// RUN: ondrix-opt %s --one-shot-bufferize="bufferize-function-boundaries function-boundary-type-conversion=identity-layout-map allow-return-allocs" --canonicalize --vectorize-ondsp-fp-filter-outputs="vector-width=4 supports-vector-fma=true interleave=4" | FileCheck %s --check-prefix=CHAINED --implicit-check-not=ondsp.reduce_mac
// RUN: not ondrix-opt %s --one-shot-bufferize="bufferize-function-boundaries function-boundary-type-conversion=identity-layout-map allow-return-allocs" --canonicalize --vectorize-ondsp-fp-filter-outputs="interleave=0" 2>&1 | FileCheck %s --check-prefix=CHAINLOW
// RUN: not ondrix-opt %s --one-shot-bufferize="bufferize-function-boundaries function-boundary-type-conversion=identity-layout-map allow-return-allocs" --canonicalize --vectorize-ondsp-fp-filter-outputs="interleave=65" 2>&1 | FileCheck %s --check-prefix=CHAINLOW

// CHAINLOW: interleave must be in [1, 64]

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
// The declaration rides the terms loop so the schedule stage keeps an
// authority for the tree rebuild; the spend record alone must never be one.
// CHECK: {ondsp.numeric = #ondsp.fp<format = f32, contract = fast>}
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
// The declared fast contract admits the rebuilt inner axis: the +0.0 seed
// stays the first leaf of chain zero, chains one to three are seeded by their
// own real products, and the pairwise merge tops out in one R-recording fold.
// CHAINED-LABEL: func.func @f32_matmul
// CHAINED: %[[C0S:.*]] = math.fma {{.*}}, %{{.*}} {ondsp.fast_used = ["fuse_multiply_add"]} : vector<4xf32>
// CHAINED: %[[C1S:.*]] = arith.mulf {{.*}} : vector<4xf32>
// CHAINED: %[[C2S:.*]] = arith.mulf {{.*}} : vector<4xf32>
// CHAINED: %[[C3S:.*]] = arith.mulf {{.*}} : vector<4xf32>
// CHAINED: %[[C0:.*]] = math.fma {{.*}}, %[[C0S]] {ondsp.fast_used = ["fuse_multiply_add"]} : vector<4xf32>
// CHAINED: %[[C1:.*]] = math.fma {{.*}}, %[[C1S]] {ondsp.fast_used = ["fuse_multiply_add"]} : vector<4xf32>
// CHAINED: %[[C2:.*]] = math.fma {{.*}}, %[[C2S]] {ondsp.fast_used = ["fuse_multiply_add"]} : vector<4xf32>
// CHAINED: %[[C3:.*]] = math.fma {{.*}}, %[[C3S]] {ondsp.fast_used = ["fuse_multiply_add"]} : vector<4xf32>
// CHAINED: %[[M0:.*]] = arith.addf %[[C0]], %[[C1]] : vector<4xf32>
// CHAINED: %[[M1:.*]] = arith.addf %[[C2]], %[[C3]] : vector<4xf32>
// CHAINED: arith.addf %[[M0]], %[[M1]] {ondsp.fast_used = ["rebuild_reduction_tree"]} : vector<4xf32>
// CHAINED: vector.store
func.func @f32_matmul(%a: tensor<4x8xf32>, %b: tensor<8x6xf32>) -> tensor<4x6xf32> {
  %c = ondrix.matmul %a, %b {
    numeric = #ondsp.fp<format = f32, contract = fast>
  } : (tensor<4x8xf32>, tensor<8x6xf32>) -> tensor<4x6xf32>
  return %c : tensor<4x6xf32>
}

// The fma contract pins the event graph: no stamp is emitted, so even with
// interleave requested the batched chain stays serial and records nothing.
// CHECK-LABEL: func.func @f32_matmul_fma
// CHECK-NOT: ondsp.numeric
// CHAINED-LABEL: func.func @f32_matmul_fma
// CHAINED-NOT: arith.mulf
// CHAINED-NOT: rebuild_reduction_tree
// CHAINED: vector.store
func.func @f32_matmul_fma(%a: tensor<4x8xf32>, %b: tensor<8x6xf32>) -> tensor<4x6xf32> {
  %c = ondrix.matmul %a, %b {
    numeric = #ondsp.fp<format = f32, contract = fma>
  } : (tensor<4x8xf32>, tensor<8x6xf32>) -> tensor<4x6xf32>
  return %c : tensor<4x6xf32>
}
