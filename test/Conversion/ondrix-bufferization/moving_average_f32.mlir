// RUN: ondrix-opt %s --one-shot-bufferize="bufferize-function-boundaries allow-return-allocs function-boundary-type-conversion=identity-layout-map" | FileCheck %s
// RUN: ondrix-opt %s --one-shot-bufferize="bufferize-function-boundaries allow-return-allocs function-boundary-type-conversion=identity-layout-map" --canonicalize --vectorize-ondsp-fp-filter-outputs="vector-width=8" | FileCheck %s --check-prefix=BATCHED

// The f32 profile bufferizes to the loop form of its declared window graph:
// seed element, left-to-right adds, one division — no unrolling, no spend.

// CHECK-LABEL: func.func @f32_moving_average
// CHECK: %[[OUT:.*]] = memref.alloc() {alignment = 64 : i64} : memref<57xf32>
// CHECK: scf.for %[[N:.*]] =
// CHECK: %[[SEED:.*]] = memref.load %{{.*}}[%[[N]]] : memref<64xf32>
// CHECK: %[[SUM:.*]] = scf.for %[[K:.*]] = %c1{{.*}} to %c8{{.*}} step %c1{{.*}} iter_args(%[[ACC:.*]] = %[[SEED]]) -> (f32)
// CHECK: %[[POS:.*]] = arith.addi %[[N]], %[[K]]
// CHECK: %[[V:.*]] = memref.load %{{.*}}[%[[POS]]]
// CHECK: arith.addf %[[ACC]], %[[V]] : f32
// CHECK: %[[MEAN:.*]] = arith.divf %[[SUM]], %{{.*}} : f32
// CHECK: memref.store %[[MEAN]], %[[OUT]][%[[N]]]

// BATCHED-LABEL: func.func @f32_moving_average
// BATCHED: vector.load {{.*}} : memref<64xf32>, vector<8xf32>
// BATCHED-COUNT-7: arith.addf {{.*}} : vector<8xf32>
// BATCHED: arith.divf {{.*}} : vector<8xf32>
// BATCHED: vector.store {{.*}} : memref<57xf32>, vector<8xf32>
// 57 outputs at width 8 leave one ordered output behind the 7 full blocks.
// BATCHED: scf.for %{{.*}} = %c56
func.func @f32_moving_average(%input: tensor<64xf32>) -> tensor<57xf32> {
  %result = ondrix.moving_average %input {
    window = 8 : i64,
    numeric = #ondsp.fp<format = f32, contract = off>
  } : (tensor<64xf32>) -> tensor<57xf32>
  return %result : tensor<57xf32>
}
