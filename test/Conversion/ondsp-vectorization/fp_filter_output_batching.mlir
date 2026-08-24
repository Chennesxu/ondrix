// RUN: ondrix-opt %s --empty-tensor-to-alloc-tensor --one-shot-bufferize="bufferize-function-boundaries allow-return-allocs function-boundary-type-conversion=identity-layout-map" --canonicalize --vectorize-ondsp-fp-filter-outputs="vector-width=8" | FileCheck %s
// RUN: ondrix-opt %s --empty-tensor-to-alloc-tensor --one-shot-bufferize="bufferize-function-boundaries allow-return-allocs function-boundary-type-conversion=identity-layout-map" --canonicalize --vectorize-ondsp-fp-filter-outputs="vector-width=8 supports-vector-fma=true" | FileCheck %s --check-prefix=FUSEDFAST

// Order-preserving f32 output batching: W independent outputs ride one
// vector, each lane running its declared event graph verbatim, and the
// remainder keeps the ordered schedule. Every contract is admitted on that
// structural ground; a shape whose extents the batched body cannot see
// statically is refused.

// The fma profile: one fused event per lane per tap, taps unrolled with the
// eight coefficient splats hoisted above the batched loop.
// CHECK-LABEL: func.func @fma_filter
// CHECK-COUNT-8: vector.splat
// CHECK: scf.for %{{.*}} = %c0{{.*}} to %c32{{.*}} step %c8
// CHECK-COUNT-8: math.fma {{.*}} : vector<8xf32>
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

// fast batches on the same order-preserving axis. Without the declared
// vector FMA it selects the separate members and spends nothing.
// CHECK-LABEL: func.func @fast_filter
// CHECK: vector.load {{.*}} : memref<40xf32>, vector<8xf32>
// CHECK: arith.mulf {{.*}} : vector<8xf32>
// CHECK: arith.addf {{.*}} : vector<8xf32>
// CHECK-NOT: math.fma
// CHECK: vector.store
// CHECK: ondsp.reduce_mac
// FUSEDFAST-NOT: rebuild_reduction_tree
// FUSEDFAST-LABEL: func.func @fast_filter
// FUSEDFAST: math.fma {{.*}} {ondsp.fast_used = ["fuse_multiply_add"]} : vector<8xf32>
// FUSEDFAST-NOT: rebuild_reduction_tree
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

// The convolution coefficient order is a reversed static subview; taps are
// scalar-loaded through it and the outputs batch on the same axis.
// CHECK-LABEL: func.func @conv_reversed_off
// CHECK: %[[REV:.*]] = memref.subview %{{.*}}[7] [8] [-1]
// CHECK: memref.load %[[REV]][%{{.*}}]
// CHECK-COUNT-8: vector.splat
// CHECK: vector.load {{.*}} : memref<64xf32>, vector<8xf32>
// CHECK: arith.mulf {{.*}} : vector<8xf32>
// CHECK: vector.store
// CHECK: ondsp.reduce_mac
func.func @conv_reversed_off(%signal: tensor<64xf32>, %kernel: tensor<8xf32>,
                             %init: tensor<57xf32>) -> tensor<57xf32> {
  %result = ondrix.conv1d %signal, %kernel, %init {
    mode = #ondrix.conv1d_mode<convolution>,
    numeric = #ondsp.fp<format = f32, contract = off>
  } : (tensor<64xf32>, tensor<8xf32>, tensor<57xf32>) -> tensor<57xf32>
  return %result : tensor<57xf32>
}

// The matmul column axis: W columns of one output row ride one vector, the
// broadcast row element is scalar-loaded, and the residual columns keep the
// ordered schedule.
// CHECK-LABEL: func.func @matmul_columns_off
// CHECK: %[[ROW:.*]] = memref.load %{{.*}} : memref<2x4xf32>
// CHECK: vector.splat %[[ROW]] : vector<8xf32>
// CHECK: scf.for %{{.*}} = %c0{{.*}} to %c16{{.*}} step %c8
// CHECK-COUNT-4: vector.load %{{.*}} : memref<4x20xf32>, vector<8xf32>
// CHECK: vector.store {{.*}} : memref<2x20xf32>, vector<8xf32>
// CHECK: scf.for %{{.*}} = %c16{{.*}} to %c20
// CHECK: arith.mulf {{.*}} : f32
func.func @matmul_columns_off(%a: tensor<2x4xf32>, %b: tensor<4x20xf32>) -> tensor<2x20xf32> {
  %r = ondrix.matmul %a, %b {
    numeric = #ondsp.fp<format = f32, contract = off>
  } : (tensor<2x4xf32>, tensor<4x20xf32>) -> tensor<2x20xf32>
  return %r : tensor<2x20xf32>
}

// A fused body batches only onto a declared vector FMA; without it the
// batching is refused (never de-fused), so the fast site keeps its scalar
// fused chain and record.
// CHECK-LABEL: func.func @matmul_columns_fast
// CHECK-NOT: vector.load
// CHECK: math.fma {{.*}} {ondsp.fast_used = ["fuse_multiply_add"]} : f32
// CHECK-NOT: vector.store
// FUSEDFAST-LABEL: func.func @matmul_columns_fast
// FUSEDFAST: math.fma {{.*}} {ondsp.fast_used = ["fuse_multiply_add"]} : vector<8xf32>
// FUSEDFAST-NOT: rebuild_reduction_tree
func.func @matmul_columns_fast(%a: tensor<2x4xf32>, %b: tensor<4x20xf32>) -> tensor<2x20xf32> {
  %r = ondrix.matmul %a, %b {
    numeric = #ondsp.fp<format = f32, contract = fast>
  } : (tensor<2x4xf32>, tensor<4x20xf32>) -> tensor<2x20xf32>
  return %r : tensor<2x20xf32>
}

// Fewer columns than lanes: no full block exists, the ordered nest stands.
// CHECK-LABEL: func.func @matmul_narrow
// CHECK-NOT: vector.load
// CHECK: arith.mulf {{.*}} : f32
func.func @matmul_narrow(%a: tensor<2x4xf32>, %b: tensor<4x6xf32>) -> tensor<2x6xf32> {
  %r = ondrix.matmul %a, %b {
    numeric = #ondsp.fp<format = f32, contract = off>
  } : (tensor<2x4xf32>, tensor<4x6xf32>) -> tensor<2x6xf32>
  return %r : tensor<2x6xf32>
}

// A spend record is discardable audit metadata, so a forged one on an exact
// fused body must never select different arithmetic: no de-fusing, ever.
// FUSEDFAST-LABEL: func.func @forged_record_stays_fused
// FUSEDFAST-NOT: arith.mulf
// FUSEDFAST: math.fma {{.*}} : vector<8xf32>
// FUSEDFAST-NOT: arith.mulf
func.func @forged_record_stays_fused(%a: memref<2x8xf32>, %b: memref<8x16xf32>, %c: memref<2x16xf32>) {
  %cst = arith.constant 0.000000e+00 : f32
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c8 = arith.constant 8 : index
  %c16 = arith.constant 16 : index
  scf.for %i = %c0 to %c2 step %c1 {
    scf.for %j = %c0 to %c16 step %c1 {
      %s = scf.for %k = %c0 to %c8 step %c1 iter_args(%acc = %cst) -> (f32) {
        %av = memref.load %a[%i, %k] : memref<2x8xf32>
        %bv = memref.load %b[%k, %j] : memref<8x16xf32>
        %n = math.fma %av, %bv, %acc {ondsp.fast_used = ["fuse_multiply_add"]} : f32
        scf.yield %n : f32
      }
      memref.store %s, %c[%i, %j] : memref<2x16xf32>
    }
  }
  return
}
