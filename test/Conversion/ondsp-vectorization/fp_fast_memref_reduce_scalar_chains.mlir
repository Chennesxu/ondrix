// RUN: ondrix-opt %s --vectorize-ondsp-fp-fast-memref-reduce="vector-width=1 interleave=4" | FileCheck %s
// RUN: ondrix-opt %s --vectorize-ondsp-fp-fast-memref-reduce="vector-width=1 interleave=4 supports-vector-fma=true" | FileCheck %s --check-prefix=FUSED
// RUN: ondrix-opt %s --vectorize-ondsp-fp-fast-memref-reduce="vector-width=1 interleave=1" | FileCheck %s --check-prefix=SINGLE

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
