// RUN: ondrix-opt %s --vectorize-ondsp-fp-fast-memref-reduce="vector-width=1 interleave=4" | FileCheck %s
// RUN: ondrix-opt %s --vectorize-ondsp-fp-fast-memref-reduce="vector-width=1 interleave=4 supports-vector-fma=true" | FileCheck %s --check-prefix=FUSED
// RUN: ondrix-opt %s --vectorize-ondsp-fp-fast-memref-reduce="vector-width=1 interleave=1" | FileCheck %s --check-prefix=SINGLE
// RUN: ondrix-opt %s --vectorize-ondsp-fp-fast-memref-reduce="vector-width=4 interleave=4" | FileCheck %s --check-prefix=LANES

// Discriminates the width-one form: scalar chains with no vector operation
// anywhere, and refusal wherever the rebuild cannot carry two chains.

// CHECK-NOT: vector

// 12 blocks of one element at 4 chains: 4 scalar seeds, a group loop of step
// 4 carrying 4 f32 iter_args, pairwise merges, one R-recording scalar fold
// seeded by the initial, and no tail loop after the fold.
// CHECK-LABEL: func.func @scalar_chains
// CHECK: %[[S0:.*]] = arith.mulf {{.*}} : f32
// CHECK: %[[S1:.*]] = arith.mulf {{.*}} : f32
// CHECK: %[[S2:.*]] = arith.mulf {{.*}} : f32
// CHECK: %[[S3:.*]] = arith.mulf {{.*}} : f32
// CHECK: %[[LOOP:.*]]:4 = scf.for {{.*}} iter_args(%[[A0:.*]] = %[[S0]], %[[A1:.*]] = %[[S1]], %[[A2:.*]] = %[[S2]], %[[A3:.*]] = %[[S3]]) -> (f32, f32, f32, f32)
// CHECK: %[[M0:.*]] = arith.addf %[[LOOP]]#0, %[[LOOP]]#1 : f32
// CHECK: %[[M1:.*]] = arith.addf %[[LOOP]]#2, %[[LOOP]]#3 : f32
// CHECK: %[[TOP:.*]] = arith.addf %[[M0]], %[[M1]] : f32
// CHECK: arith.addf %{{.*}}, %[[TOP]] {ondsp.fast_used = ["rebuild_reduction_tree"]} : f32
// CHECK-NOT: scf.for
func.func @scalar_chains(%lhs: memref<12xf32>, %rhs: memref<12xf32>, %init: f32) -> f32 {
  %r = ondsp.reduce_mac %init, %lhs, %rhs {numeric = #ondsp.fp<format = f32, contract = fast>} : (f32, memref<12xf32>, memref<12xf32>) -> f32
  return %r : f32
}

// FUSED-LABEL: func.func @scalar_chains
// FUSED-COUNT-4: arith.mulf {{.*}} : f32
// FUSED: scf.for
// FUSED-COUNT-4: math.fma {{.*}} {ondsp.fast_used = ["fuse_multiply_add"]} : f32

// One effective chain at width one would spend R on the ordered schedule.
// SINGLE-LABEL: func.func @scalar_chains
// SINGLE: ondsp.reduce_mac

// A dynamic extent has no compile-time block count to split into chains.
// CHECK-LABEL: func.func @dynamic_refused
// CHECK: ondsp.reduce_mac
func.func @dynamic_refused(%lhs: memref<?xf32>, %rhs: memref<?xf32>, %init: f32) -> f32 {
  %r = ondsp.reduce_mac %init, %lhs, %rhs {numeric = #ondsp.fp<format = f32, contract = fast>} : (f32, memref<?xf32>, memref<?xf32>) -> f32
  return %r : f32
}

// Bufferization hands the reduction its operands through an extent-erasing
// cast, so reading the operand type alone refuses a static twelve.
// CHECK-LABEL: func.func @cast_erased_extent
// CHECK-COUNT-4: arith.mulf {{.*}} : f32
// CHECK: arith.addf %{{.*}}, %{{.*}} {ondsp.fast_used = ["rebuild_reduction_tree"]} : f32
func.func @cast_erased_extent(%lhs: memref<12xf32>, %rhs: memref<12xf32>, %init: f32) -> f32 {
  %l = memref.cast %lhs : memref<12xf32> to memref<?xf32>
  %r = memref.cast %rhs : memref<12xf32> to memref<?xf32>
  %s = ondsp.reduce_mac %init, %l, %r {numeric = #ondsp.fp<format = f32, contract = fast>} : (f32, memref<?xf32>, memref<?xf32>) -> f32
  return %s : f32
}

// A convolution kernel is read backwards. One lane loads through the memref's
// own stride, so contiguity is the Vector lowering's obligation, not this
// rewrite's; at four lanes the same reduction stays ordered.
// CHECK-LABEL: func.func @reversed_kernel
// CHECK-COUNT-4: arith.mulf {{.*}} : f32
// CHECK: arith.addf %{{.*}}, %{{.*}} {ondsp.fast_used = ["rebuild_reduction_tree"]} : f32
// LANES-LABEL: func.func @reversed_kernel
// LANES: ondsp.reduce_mac
func.func @reversed_kernel(%lhs: memref<12xf32>, %rhs: memref<12xf32>, %init: f32) -> f32 {
  %v = memref.subview %rhs[11] [12] [-1] : memref<12xf32> to memref<12xf32, strided<[-1], offset: 11>>
  %s = ondsp.reduce_mac %init, %lhs, %v {numeric = #ondsp.fp<format = f32, contract = fast>} : (f32, memref<12xf32>, memref<12xf32, strided<[-1], offset: 11>>) -> f32
  return %s : f32
}
