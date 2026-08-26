// RUN: ondrix-opt %s --one-shot-bufferize="bufferize-function-boundaries function-boundary-type-conversion=identity-layout-map allow-return-allocs" | FileCheck %s
// RUN: ondrix-opt %s --one-shot-bufferize="bufferize-function-boundaries function-boundary-type-conversion=identity-layout-map allow-return-allocs" --canonicalize --vectorize-ondsp-fp-filter-outputs="vector-width=4 supports-vector-fma=true" | FileCheck %s --check-prefix=BATCHED
// RUN: ondrix-opt %s --one-shot-bufferize="bufferize-function-boundaries function-boundary-type-conversion=identity-layout-map allow-return-allocs" --canonicalize --vectorize-ondsp-fp-filter-outputs="vector-width=4 supports-vector-fma=false" | FileCheck %s --check-prefix=NOFMA
// RUN: ondrix-opt %s --one-shot-bufferize="bufferize-function-boundaries function-boundary-type-conversion=identity-layout-map allow-return-allocs" --canonicalize --vectorize-ondsp-fp-filter-outputs="vector-width=16 supports-vector-fma=true" | FileCheck %s --check-prefix=TOOWIDE
// RUN: ondrix-opt %s --one-shot-bufferize="bufferize-function-boundaries function-boundary-type-conversion=identity-layout-map allow-return-allocs" --canonicalize --vectorize-ondsp-fp-filter-outputs="vector-width=3 supports-vector-fma=true" | FileCheck %s --check-prefix=PARTIAL

// The table is transposed, `[term][output]`, so a tile of outputs is one
// contiguous load; rank two is admissible because the binary32 profile carries
// no prefix proof to resolve back through rank-reducing views.
// CHECK-DAG: memref.global "private" constant @__ondrix_dct8_f32 : memref<8x8xf32>

// CHECK-LABEL: func.func @f32_dct8
// CHECK: %[[OUT:.*]] = memref.alloc() {alignment = 64 : i64} : memref<8xf32>
// CHECK: %[[TABLE:.*]] = memref.get_global @__ondrix_dct8_f32
// CHECK: scf.for %[[K:.*]] = %{{.*}} to %{{.*}} step %{{.*}} {
// Each output starts AT its first product; seeding the additive identity
// instead would export +0.0 where the contract exports -0.0.
// CHECK: %[[X0:.*]] = memref.load %{{.*}}[%{{.*}}] : memref<8xf32>
// CHECK: %[[C0:.*]] = memref.load %[[TABLE]][%{{.*}}, %[[K]]] : memref<8x8xf32>
// CHECK: %[[SEED:.*]] = arith.mulf %[[X0]], %[[C0]] : f32
// CHECK: %[[SUM:.*]] = scf.for %[[N:.*]] = %{{.*}} to %{{.*}} step %{{.*}} iter_args(%[[ACC:.*]] = %[[SEED]]) -> (f32) {
// CHECK: memref.load %{{.*}}[%[[N]]] : memref<8xf32>
// CHECK: memref.load %[[TABLE]][%[[N]], %[[K]]] : memref<8x8xf32>
// CHECK: memref.store %[[SUM]], %[[OUT]][%[[K]]]

// Four outputs share one pass over the terms, each lane keeping its own first
// product as its seed and the same terms in the same order, so the per-output
// event graph is unchanged and no permission is recorded.
// BATCHED-LABEL: func.func @f32_dct8
// BATCHED: %[[SPLAT:.*]] = vector.splat %{{.*}} : vector<4xf32>
// BATCHED: scf.for
// BATCHED: %[[W0:.*]] = vector.load %{{.*}}[%{{.*}}, %{{.*}}] : memref<8x8xf32>, vector<4xf32>
// BATCHED: %[[LANES:.*]] = arith.mulf %[[SPLAT]], %[[W0]] : vector<4xf32>
// BATCHED: arith.mulf {{.*}} : vector<4xf32>
// BATCHED: arith.addf %[[LANES]], {{.*}} : vector<4xf32>
// BATCHED: vector.store {{.*}} : memref<8xf32>, vector<4xf32>
// BATCHED-NOT: ondsp.fast_used

func.func @f32_dct8(%input: tensor<8xf32>) -> tensor<8xf32> {
  %result = ondrix.dct %input {
    input_numeric = #ondsp.fp<format = f32, contract = off>,
    output_numeric = #ondsp.fp<format = f32, contract = off>
  } : (tensor<8xf32>) -> tensor<8xf32>
  return %result : tensor<8xf32>
}

func.func @f32_dct8_fast(%input: tensor<8xf32>) -> tensor<8xf32> {
  %result = ondrix.dct %input {
    input_numeric = #ondsp.fp<format = f32, contract = fast>,
    output_numeric = #ondsp.fp<format = f32, contract = fast>
  } : (tensor<8xf32>) -> tensor<8xf32>
  return %result : tensor<8xf32>
}

// A fused body batches as fused events, never de-fused ones.
// BATCHED-LABEL: func.func @f32_dct8_fast
// BATCHED: math.fma {{.*}} : vector<4xf32>

// Without a declared vector fused multiply-add there is no fused lane to batch
// onto, and de-fusing would need an authority this loop no longer carries, so
// the fused profile keeps its ordered scalar form.
// NOFMA-LABEL: func.func @f32_dct8_fast
// NOFMA-NOT: vector<4xf32>
// NOFMA: math.fma {{.*}} : f32

// Eight outputs at width three: two full blocks, then outputs six and seven
// keep the ordered scalar loop, whose lower bound is the batched end.
// PARTIAL-LABEL: func.func @f32_dct8
// PARTIAL-DAG: %[[END:.*]] = arith.constant 6 : index
// PARTIAL-DAG: %[[STEP:.*]] = arith.constant 3 : index
// PARTIAL: scf.for %{{.*}} = %{{.*}} to %[[END]] step %[[STEP]] {
// PARTIAL: vector.store {{.*}} : memref<8xf32>, vector<3xf32>
// PARTIAL: scf.for %[[K:.*]] = %[[END]] to %{{.*}} step %{{.*}} {
// PARTIAL: memref.store {{.*}}, %{{.*}}[%[[K]]] : memref<8xf32>

// A tile wider than the whole output axis has no full block to fill, so the
// batching refuses rather than emitting an empty batched loop beside the
// ordered one.
// TOOWIDE-LABEL: func.func @f32_dct8
// TOOWIDE-NOT: vector.store
// TOOWIDE: memref.store {{.*}} : memref<8xf32>
