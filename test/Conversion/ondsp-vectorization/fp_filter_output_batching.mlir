// RUN: ondrix-opt %s --empty-tensor-to-alloc-tensor --one-shot-bufferize="bufferize-function-boundaries function-boundary-type-conversion=identity-layout-map" --canonicalize --vectorize-ondsp-fp-filter-outputs="vector-width=8" | FileCheck %s

// Order-preserving f32 output batching under the exact contracts: W
// independent outputs ride one vector, each lane running its declared event
// graph verbatim, and the remainder keeps the ordered schedule. The fast
// contract is refused — it relaxes the correctness relation instead of
// pinning the event graph and has its own horizontal mode — as is any shape
// whose extents the batched body would need but cannot see statically.

// The fma profile: one fused event per lane per tap, and the single-output
// scalar recurrence is replaced by a per-lane vector recurrence.
// CHECK-LABEL: func.func @fma_filter
// CHECK: scf.for %{{.*}} = %c0{{.*}} to %c32{{.*}} step %c8
// CHECK: scf.for %{{.*}} iter_args(%{{.*}} = %{{.*}}) -> (vector<8xf32>)
// CHECK: vector.load {{.*}} : memref<40xf32>, vector<8xf32>
// CHECK: vector.splat
// CHECK: math.fma {{.*}} : vector<8xf32>
// CHECK-NOT: arith.mulf
// CHECK: vector.store {{.*}} : memref<33xf32>, vector<8xf32>
// CHECK: scf.for %{{.*}} = %c32{{.*}} to %c33
// CHECK: ondsp.reduce_mac
func.func @fma_filter(%input: tensor<40xf32>, %coeffs: tensor<8xf32>,
                      %init: tensor<33xf32>) -> tensor<33xf32> {
  %result = ondrix.fir_filter %input, %coeffs, %init {
    boundary = #ondrix.fir_boundary<valid>,
    numeric = #ondsp.fp<format = f32, contract = fma>
  } : (tensor<40xf32>, tensor<8xf32>, tensor<33xf32>) -> tensor<33xf32>
  return %result : tensor<33xf32>
}

// The off profile: separate ordered product and accumulation events per lane,
// with no fused operation anywhere in the batched body.
// CHECK-LABEL: func.func @off_filter
// CHECK: vector.load {{.*}} : memref<40xf32>, vector<8xf32>
// CHECK: arith.mulf {{.*}} : vector<8xf32>
// CHECK: arith.addf {{.*}} : vector<8xf32>
// CHECK-NOT: math.fma
// CHECK: vector.store
// CHECK: ondsp.reduce_mac
func.func @off_filter(%input: tensor<40xf32>, %coeffs: tensor<8xf32>,
                      %init: tensor<33xf32>) -> tensor<33xf32> {
  %result = ondrix.fir_filter %input, %coeffs, %init {
    boundary = #ondrix.fir_boundary<valid>,
    numeric = #ondsp.fp<format = f32, contract = off>
  } : (tensor<40xf32>, tensor<8xf32>, tensor<33xf32>) -> tensor<33xf32>
  return %result : tensor<33xf32>
}

// fast stays on the ordered schedule: no batched loop, no vector operations.
// CHECK-LABEL: func.func @fast_filter
// CHECK-NOT: vector.load
// CHECK: ondsp.reduce_mac
// CHECK-NOT: vector.store
func.func @fast_filter(%input: tensor<40xf32>, %coeffs: tensor<8xf32>,
                       %init: tensor<33xf32>) -> tensor<33xf32> {
  %result = ondrix.fir_filter %input, %coeffs, %init {
    boundary = #ondrix.fir_boundary<valid>,
    numeric = #ondsp.fp<format = f32, contract = fast>
  } : (tensor<40xf32>, tensor<8xf32>, tensor<33xf32>) -> tensor<33xf32>
  return %result : tensor<33xf32>
}

// Fewer outputs than lanes: no full block exists, the ordered loop stands.
// CHECK-LABEL: func.func @narrow_filter
// CHECK-NOT: vector.load
// CHECK: ondsp.reduce_mac
func.func @narrow_filter(%input: tensor<11xf32>, %coeffs: tensor<8xf32>,
                         %init: tensor<4xf32>) -> tensor<4xf32> {
  %result = ondrix.fir_filter %input, %coeffs, %init {
    boundary = #ondrix.fir_boundary<valid>,
    numeric = #ondsp.fp<format = f32, contract = fma>
  } : (tensor<11xf32>, tensor<8xf32>, tensor<4xf32>) -> tensor<4xf32>
  return %result : tensor<4xf32>
}

// Dynamic extents: the batched body cannot see its bounds, so the loop is
// refused rather than partially understood.
// CHECK-LABEL: func.func @dynamic_filter
// CHECK-NOT: vector.load
// CHECK: ondsp.reduce_mac
func.func @dynamic_filter(%input: tensor<?xf32>, %coeffs: tensor<?xf32>,
                          %init: tensor<?xf32>) -> tensor<?xf32> {
  %result = ondrix.fir_filter %input, %coeffs, %init {
    boundary = #ondrix.fir_boundary<valid>,
    numeric = #ondsp.fp<format = f32, contract = fma>
  } : (tensor<?xf32>, tensor<?xf32>, tensor<?xf32>) -> tensor<?xf32>
  return %result : tensor<?xf32>
}
