// RUN: ondrix-opt %s --one-shot-bufferize="bufferize-function-boundaries allow-return-allocs function-boundary-type-conversion=identity-layout-map" | FileCheck %s
// RUN: ondrix-opt %s --one-shot-bufferize="bufferize-function-boundaries allow-return-allocs function-boundary-type-conversion=identity-layout-map" --canonicalize --vectorize-ondsp-fp-filter-outputs="vector-width=8" | FileCheck %s --check-prefix=BATCHED
// RUN: ondrix-opt %s --one-shot-bufferize="bufferize-function-boundaries allow-return-allocs function-boundary-type-conversion=identity-layout-map" --canonicalize --vectorize-ondsp-fp-filter-outputs="vector-width=4 interleave=4" | FileCheck %s --check-prefix=CHAINED

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
// An off window carries no declaration, so the stamp must not appear.
// CHECK-NOT: ondsp.numeric
// CHECK: %[[MEAN:.*]] = arith.divf %[[SUM]], %{{.*}} : f32
// CHECK: memref.store %[[MEAN]], %[[OUT]][%[[N]]]

// BATCHED-LABEL: func.func @f32_moving_average
// BATCHED: vector.load {{.*}} : memref<64xf32>, vector<8xf32>
// BATCHED-COUNT-7: arith.addf {{.*}} : vector<8xf32>
// BATCHED: arith.divf {{.*}} : vector<8xf32>
// BATCHED: vector.store {{.*}} : memref<57xf32>, vector<8xf32>
// 57 outputs at width 8 leave one ordered output behind the 7 full blocks.
// BATCHED: scf.for %{{.*}} = %c56

// The off window admits only the order-preserving batch: no chain rebuild,
// no record, even with interleave requested.
// CHAINED-LABEL: func.func @f32_moving_average
// CHAINED-NOT: ondsp.fast_used
func.func @f32_moving_average(%input: tensor<64xf32>) -> tensor<57xf32> {
  %result = ondrix.moving_average %input {
    window = 8 : i64,
    numeric = #ondsp.fp<format = f32, contract = off>
  } : (tensor<64xf32>) -> tensor<57xf32>
  return %result : tensor<57xf32>
}

// The declaration rides the window loop so the schedule stage keeps an
// authority for the tree rebuild; the spend record alone must never be one.
// CHECK-LABEL: func.func @f32_moving_average_fast
// CHECK: scf.for
// CHECK: {ondsp.numeric = #ondsp.fp<format = f32, contract = fast>}

// Without an interleave the batched fast window keeps the serial per-lane
// fold and records nothing.
// BATCHED-LABEL: func.func @f32_moving_average_fast
// BATCHED-NOT: ondsp.fast_used
// BATCHED: arith.divf {{.*}} : vector<8xf32>

// The declared fast contract admits the rebuilt window: two chains seeded
// by the leading loads, alternating terms, one R-recording top fold.
// CHAINED-LABEL: func.func @f32_moving_average_fast
// CHAINED: %[[T0:.*]] = vector.load {{.*}} : memref<16xf32>, vector<4xf32>
// CHAINED: %[[T1:.*]] = vector.load {{.*}} : memref<16xf32>, vector<4xf32>
// CHAINED: %[[T2:.*]] = vector.load {{.*}} : memref<16xf32>, vector<4xf32>
// CHAINED: %[[A0:.*]] = arith.addf %[[T0]], %[[T2]] : vector<4xf32>
// CHAINED: %[[T3:.*]] = vector.load {{.*}} : memref<16xf32>, vector<4xf32>
// CHAINED: %[[B0:.*]] = arith.addf %[[T1]], %[[T3]] : vector<4xf32>
// CHAINED: %[[T4:.*]] = vector.load {{.*}} : memref<16xf32>, vector<4xf32>
// CHAINED: %[[A1:.*]] = arith.addf %[[A0]], %[[T4]] : vector<4xf32>
// CHAINED: %[[T5:.*]] = vector.load {{.*}} : memref<16xf32>, vector<4xf32>
// CHAINED: %[[B1:.*]] = arith.addf %[[B0]], %[[T5]] : vector<4xf32>
// CHAINED: %[[T6:.*]] = vector.load {{.*}} : memref<16xf32>, vector<4xf32>
// CHAINED: %[[A2:.*]] = arith.addf %[[A1]], %[[T6]] : vector<4xf32>
// CHAINED: %[[T7:.*]] = vector.load {{.*}} : memref<16xf32>, vector<4xf32>
// CHAINED: %[[B2:.*]] = arith.addf %[[B1]], %[[T7]] : vector<4xf32>
// CHAINED: arith.addf %[[A2]], %[[B2]] {ondsp.fast_used = ["rebuild_reduction_tree"]} : vector<4xf32>
// CHAINED: arith.divf {{.*}} : vector<4xf32>
// CHAINED: vector.store {{.*}} : memref<9xf32>, vector<4xf32>
// 9 outputs at width 4 leave one ordered output behind the 2 full blocks.
// CHAINED: scf.for %{{.*}} = %c8
func.func @f32_moving_average_fast(%input: tensor<16xf32>) -> tensor<9xf32> {
  %result = ondrix.moving_average %input {
    window = 8 : i64,
    numeric = #ondsp.fp<format = f32, contract = fast>
  } : (tensor<16xf32>) -> tensor<9xf32>
  return %result : tensor<9xf32>
}

// Below four terms every chained tree is the declared left fold itself, so
// the fast window stays ordered and no record is written.
// CHAINED-LABEL: func.func @f32_moving_average_fast_short
// CHAINED-NOT: ondsp.fast_used
func.func @f32_moving_average_fast_short(%input: tensor<8xf32>) -> tensor<6xf32> {
  %result = ondrix.moving_average %input {
    window = 3 : i64,
    numeric = #ondsp.fp<format = f32, contract = fast>
  } : (tensor<8xf32>) -> tensor<6xf32>
  return %result : tensor<6xf32>
}
