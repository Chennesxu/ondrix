// RUN: ondrix-opt %s --vectorize-ondsp-fp-fast-memref-reduce="vector-width=4 interleave=4" | FileCheck %s
// RUN: ondrix-opt %s --vectorize-ondsp-fp-fast-memref-reduce="vector-width=4 interleave=4 supports-vector-fma=true" | FileCheck %s --check-prefix=FUSED
// RUN: ondrix-opt %s --vectorize-ondsp-fp-fast-memref-reduce="vector-width=4 interleave=3" | FileCheck %s --check-prefix=ODD
// RUN: not ondrix-opt %s --vectorize-ondsp-fp-fast-memref-reduce="interleave=0" 2>&1 | FileCheck %s --check-prefix=LOW
// RUN: not ondrix-opt %s --vectorize-ondsp-fp-fast-memref-reduce="interleave=65" 2>&1 | FileCheck %s --check-prefix=HIGH

// Discriminates chain count, group step, leftover-block routing, and the
// pairwise merge from the single-chain rebuild the base test pins.

// LOW: interleave must be in [1, 64]
// HIGH: interleave must be in [1, 64]

// CHECK-NOT: fastmath
// CHECK: module attributes {ondsp.fast_used = ["rebuild_reduction_tree"]}

// 64/4 = 16 blocks, 4 chains, no leftover: 4 seeded chains, one group loop of
// step 16 carrying 4 iter_args, 3 pairwise merges, one R-recording fold.
// CHECK-LABEL: func.func @chains_divide_blocks
// CHECK-NOT: arith.constant dense
// CHECK: %[[S0:.*]] = arith.mulf {{.*}} : vector<4xf32>
// CHECK: %[[S1:.*]] = arith.mulf {{.*}} : vector<4xf32>
// CHECK: %[[S2:.*]] = arith.mulf {{.*}} : vector<4xf32>
// CHECK: %[[S3:.*]] = arith.mulf {{.*}} : vector<4xf32>
// CHECK: %[[STEP:.*]] = arith.constant 16 : index
// CHECK: %[[LOOP:.*]]:4 = scf.for {{.*}} step %[[STEP]] iter_args(%[[A0:.*]] = %[[S0]], %[[A1:.*]] = %[[S1]], %[[A2:.*]] = %[[S2]], %[[A3:.*]] = %[[S3]])
// CHECK: arith.addf %[[A0]], %{{.*}} : vector<4xf32>
// CHECK: arith.addf %[[A1]], %{{.*}} : vector<4xf32>
// CHECK: arith.addf %[[A2]], %{{.*}} : vector<4xf32>
// CHECK: arith.addf %[[A3]], %{{.*}} : vector<4xf32>
// CHECK: %[[M0:.*]] = arith.addf %[[LOOP]]#0, %[[LOOP]]#1 : vector<4xf32>
// CHECK: %[[M1:.*]] = arith.addf %[[LOOP]]#2, %[[LOOP]]#3 : vector<4xf32>
// CHECK: %[[TOP:.*]] = arith.addf %[[M0]], %[[M1]] : vector<4xf32>
// CHECK: vector.shuffle %[[TOP]], %[[TOP]] [0, 1] : vector<4xf32>, vector<4xf32>
// CHECK: vector.shuffle %[[TOP]], %[[TOP]] [2, 3] : vector<4xf32>, vector<4xf32>
// CHECK: arith.addf {{.*}} {ondsp.fast_used = ["rebuild_reduction_tree"]} : f32
// The fused capability spends F per chained term, on the loop terms and on
// the leftover blocks alike; the seeds stay raw products (nothing to fuse).
// FUSED: module attributes {ondsp.fast_used = ["fuse_multiply_add", "rebuild_reduction_tree"]}
// FUSED-LABEL: func.func @chains_divide_blocks
// FUSED-COUNT-4: arith.mulf {{.*}} : vector<4xf32>
// FUSED: scf.for
// FUSED-COUNT-4: math.fma {{.*}} {ondsp.fast_used = ["fuse_multiply_add"]} : vector<4xf32>
// FUSED: scf.yield
// FUSED: arith.addf {{.*}} {ondsp.fast_used = ["rebuild_reduction_tree"]} : f32
func.func @chains_divide_blocks(%lhs: memref<64xf32>, %rhs: memref<64xf32>, %init: f32) -> f32 {
  %r = ondsp.reduce_mac %init, %lhs, %rhs {numeric = #ondsp.fp<format = f32, contract = fast>} : (f32, memref<64xf32>, memref<64xf32>) -> f32
  return %r : f32
}

// 29/4 = 7 blocks at 3 chains: one group of 3, leftover (7-3)%3 = 1, the odd
// third chain rides through the pairwise merge, and one scalar tail element.
// ODD-LABEL: func.func @odd_chain_merge
// ODD: %[[OLOOP:.*]]:3 = scf.for
// ODD: scf.yield
// ODD: %[[OL0:.*]] = arith.mulf {{.*}} : vector<4xf32>
// ODD-NEXT: %[[OC0:.*]] = arith.addf %[[OLOOP]]#0, %[[OL0]] : vector<4xf32>
// ODD: %[[OM0:.*]] = arith.addf %[[OC0]], %[[OLOOP]]#1 : vector<4xf32>
// ODD: %[[OTOP:.*]] = arith.addf %[[OM0]], %[[OLOOP]]#2 : vector<4xf32>
// ODD: vector.shuffle %[[OTOP]], %[[OTOP]] [0, 1]
// ODD: scf.for {{.*}} -> (f32)
func.func @odd_chain_merge(%lhs: memref<29xf32>, %rhs: memref<29xf32>, %init: f32) -> f32 {
  %r = ondsp.reduce_mac %init, %lhs, %rhs {numeric = #ondsp.fp<format = f32, contract = fast>} : (f32, memref<29xf32>, memref<29xf32>) -> f32
  return %r : f32
}

// 44/4 = 11 blocks, 4 chains, leftover (11-4)%4 = 3 full blocks folded one
// per chain after the loop, then the scalar tail (44 = 11*4, none) is empty.
// CHECK-LABEL: func.func @leftover_blocks_fold_per_chain
// CHECK: %[[LLOOP:.*]]:4 = scf.for
// CHECK: scf.yield
// CHECK: %[[L0:.*]] = arith.mulf {{.*}} : vector<4xf32>
// CHECK-NEXT: arith.addf %[[LLOOP]]#0, %[[L0]] : vector<4xf32>
// CHECK: arith.addf %[[LLOOP]]#1, %{{.*}} : vector<4xf32>
// CHECK: arith.addf %[[LLOOP]]#2, %{{.*}} : vector<4xf32>
// CHECK: arith.addf {{.*}} {ondsp.fast_used = ["rebuild_reduction_tree"]} : f32
func.func @leftover_blocks_fold_per_chain(%lhs: memref<44xf32>, %rhs: memref<44xf32>, %init: f32) -> f32 {
  %r = ondsp.reduce_mac %init, %lhs, %rhs {numeric = #ondsp.fp<format = f32, contract = fast>} : (f32, memref<44xf32>, memref<44xf32>) -> f32
  return %r : f32
}

// 8/4 = 2 blocks clamp the 4 requested chains to 2.
// CHECK-LABEL: func.func @chains_clamped_to_blocks
// CHECK: %[[C0:.*]] = arith.mulf {{.*}} : vector<4xf32>
// CHECK: %[[C1:.*]] = arith.mulf {{.*}} : vector<4xf32>
// CHECK: scf.for {{.*}} iter_args(%{{.*}} = %[[C0]], %{{.*}} = %[[C1]])
// CHECK: arith.addf
// CHECK: arith.addf {{.*}} {ondsp.fast_used = ["rebuild_reduction_tree"]} : f32
func.func @chains_clamped_to_blocks(%lhs: memref<8xf32>, %rhs: memref<8xf32>, %init: f32) -> f32 {
  %r = ondsp.reduce_mac %init, %lhs, %rhs {numeric = #ondsp.fp<format = f32, contract = fast>} : (f32, memref<8xf32>, memref<8xf32>) -> f32
  return %r : f32
}

// A dynamic extent has no compile-time block count: single chain kept.
// CHECK-LABEL: func.func @dynamic_keeps_single_chain
// CHECK: scf.if
// CHECK: scf.for {{.*}} iter_args(%{{.*}} = %{{.*}}) -> (vector<4xf32>)
// CHECK-NOT: {{.*}}:2 = scf.for
func.func @dynamic_keeps_single_chain(%lhs: memref<?xf32>, %rhs: memref<?xf32>, %init: f32) -> f32 {
  %r = ondsp.reduce_mac %init, %lhs, %rhs {numeric = #ondsp.fp<format = f32, contract = fast>} : (f32, memref<?xf32>, memref<?xf32>) -> f32
  return %r : f32
}
