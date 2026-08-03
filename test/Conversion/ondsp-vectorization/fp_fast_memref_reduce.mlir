// RUN: ondrix-opt %s --vectorize-ondsp-fp-fast-memref-reduce | FileCheck %s
// RUN: ondrix-opt %s --vectorize-ondsp-fp-fast-memref-reduce="supports-vector-fma=true" | FileCheck %s --check-prefix=FUSED
// RUN: not ondrix-opt %s --vectorize-ondsp-fp-fast-memref-reduce="vector-width=1" 2>&1 | FileCheck %s --check-prefix=WIDTH
// RUN: not ondrix-opt %s --vectorize-ondsp-fp-fast-memref-reduce="vector-width=5000" 2>&1 | FileCheck %s --check-prefix=WIDE

// Structure gate for the term-conserving rebuild; the pass description
// carries the authorization argument. The whole-file NOT is the leak gate:
// the schedule is the selection, so any flag reaching here would mean the
// choice had been handed to the backend instead.

// WIDTH: vector-width must be greater than one
// WIDE: vector-width must not exceed 4096

// CHECK-NOT: fastmath

// CHECK-LABEL: func.func @f32_dot_fast_dynamic
// CHECK: cf.assert {{.*}}equal operand lengths
// CHECK: %[[BLOCKS:.*]] = arith.subi
// CHECK: %[[HAS:.*]] = arith.cmpi ugt, %[[BLOCKS]], %{{.*}}
// CHECK: scf.if %[[HAS]] -> (f32) {
// The lane seed is data: W real products, no synthesized identity vector.
// CHECK-NOT: arith.constant dense
// CHECK: %[[SL:.*]] = vector.load {{.*}} : memref<?xf32>, vector<8xf32>
// CHECK: %[[SR:.*]] = vector.load {{.*}} : memref<?xf32>, vector<8xf32>
// CHECK: %[[SEED:.*]] = arith.mulf %[[SL]], %[[SR]] : vector<8xf32>
// CHECK: %[[PARTIAL:.*]] = scf.for {{.*}} iter_args(%[[ACC:.*]] = %[[SEED]]) -> (vector<8xf32>)
// CHECK: %[[P:.*]] = arith.mulf %{{.*}}, %{{.*}} : vector<8xf32>
// CHECK: arith.addf %[[ACC]], %[[P]] : vector<8xf32>
// CHECK: %[[REDUCED:.*]] = vector.reduction <add>, %[[PARTIAL]] : vector<8xf32> into f32
// CHECK: %[[TAIL:.*]] = scf.for {{.*}} iter_args({{.*}} = %[[REDUCED]]) -> (f32)
// CHECK: arith.addf %[[INIT:.*]], %[[TAIL]] : f32
// Fewer than W elements has no lane to fill, so the ordered schedule is kept
// rather than padded up to one block.
// CHECK: } else {
// CHECK: scf.for {{.*}} iter_args({{.*}} = %[[INIT]]) -> (f32)
func.func @f32_dot_fast_dynamic(%lhs: memref<?xf32>, %rhs: memref<?xf32>, %init: f32) -> f32 {
  %r = ondsp.reduce_mac %init, %lhs, %rhs {numeric = #ondsp.fp<format = f32, contract = fast>} : (f32, memref<?xf32>, memref<?xf32>) -> f32
  return %r : f32
}

// A declared vector FMA changes only which member of the legal set runs. The
// seed stays a raw product because it has no addend to fuse: selection is
// per term, not uniform across the reduction.
// FUSED-LABEL: func.func @f32_dot_fast_dynamic
// FUSED: %[[FSEED:.*]] = arith.mulf %{{.*}}, %{{.*}} : vector<8xf32>
// FUSED: scf.for {{.*}} iter_args(%[[FACC:.*]] = %[[FSEED]]) -> (vector<8xf32>)
// FUSED: math.fma %{{.*}}, %{{.*}}, %[[FACC]] : vector<8xf32>
// FUSED-NOT: fastmath

// CHECK-LABEL: func.func @f32_dot_fast_static
// CHECK: vector.reduction <add>
func.func @f32_dot_fast_static(%lhs: memref<40xf32>, %rhs: memref<40xf32>) -> f32 {
  %zero = arith.constant 0.0 : f32
  %r = ondsp.reduce_mac %zero, %lhs, %rhs {numeric = #ondsp.fp<format = f32, contract = fast>} : (f32, memref<40xf32>, memref<40xf32>) -> f32
  return %r : f32
}

// The exact contracts pin the ordered event graph; regrouping is refused.
// CHECK-LABEL: func.func @f32_dot_fma_kept
// CHECK-NOT: vector.load
// CHECK: ondsp.reduce_mac
func.func @f32_dot_fma_kept(%lhs: memref<?xf32>, %rhs: memref<?xf32>) -> f32 {
  %zero = arith.constant 0.0 : f32
  %r = ondsp.reduce_mac %zero, %lhs, %rhs {numeric = #ondsp.fp<format = f32, contract = fma>} : (f32, memref<?xf32>, memref<?xf32>) -> f32
  return %r : f32
}

// CHECK-LABEL: func.func @f32_dot_off_kept
// CHECK-NOT: vector.load
// CHECK: ondsp.reduce_mac
func.func @f32_dot_off_kept(%lhs: memref<?xf32>, %rhs: memref<?xf32>) -> f32 {
  %zero = arith.constant 0.0 : f32
  %r = ondsp.reduce_mac %zero, %lhs, %rhs {numeric = #ondsp.fp<format = f32, contract = off>} : (f32, memref<?xf32>, memref<?xf32>) -> f32
  return %r : f32
}

// A non-unit minor stride stays on the ordered scalar path.
// CHECK-LABEL: func.func @f32_dot_fast_strided_kept
// CHECK-NOT: vector.load
// CHECK: ondsp.reduce_mac
func.func @f32_dot_fast_strided_kept(%lhs: memref<20xf32, strided<[2]>>,
                                     %rhs: memref<20xf32>) -> f32 {
  %zero = arith.constant 0.0 : f32
  %r = ondsp.reduce_mac %zero, %lhs, %rhs {numeric = #ondsp.fp<format = f32, contract = fast>} : (f32, memref<20xf32, strided<[2]>>, memref<20xf32>) -> f32
  return %r : f32
}
