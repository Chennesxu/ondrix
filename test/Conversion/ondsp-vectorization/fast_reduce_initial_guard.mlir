// RUN: ondrix-opt %s --vectorize-ondsp-fp-fast-memref-reduce=vector-width=8 | FileCheck %s

// The rewrite seeds its lanes with synthesized +0.0 and adds the operation's
// initial value back at the end, which is only value-neutral for canonical
// +0.0. With initial -0.0 and one term the batched result is +0.0 where both
// declared graphs give -0.0, and one term admits no reassociation. Anything
// but a statically canonical +0.0 keeps the ordered form.

// CHECK-LABEL: func.func @negative_zero_initial
// CHECK: ondsp.reduce_mac
// CHECK-NOT: vector.reduction
func.func @negative_zero_initial(%lhs: memref<16xf32>, %rhs: memref<16xf32>) -> f32 {
  %initial = arith.constant -0.0 : f32
  %result = ondsp.reduce_mac %initial, %lhs, %rhs {
    numeric = #ondsp.fp<format = f32, contract = fast>
  } : (f32, memref<16xf32>, memref<16xf32>) -> f32
  return %result : f32
}

// CHECK-LABEL: func.func @nonzero_initial
// CHECK: ondsp.reduce_mac
// CHECK-NOT: vector.reduction
func.func @nonzero_initial(%lhs: memref<16xf32>, %rhs: memref<16xf32>) -> f32 {
  %initial = arith.constant 1.0 : f32
  %result = ondsp.reduce_mac %initial, %lhs, %rhs {
    numeric = #ondsp.fp<format = f32, contract = fast>
  } : (f32, memref<16xf32>, memref<16xf32>) -> f32
  return %result : f32
}

// CHECK-LABEL: func.func @dynamic_initial
// CHECK: ondsp.reduce_mac
// CHECK-NOT: vector.reduction
func.func @dynamic_initial(%initial: f32, %lhs: memref<16xf32>, %rhs: memref<16xf32>) -> f32 {
  %result = ondsp.reduce_mac %initial, %lhs, %rhs {
    numeric = #ondsp.fp<format = f32, contract = fast>
  } : (f32, memref<16xf32>, memref<16xf32>) -> f32
  return %result : f32
}

// CHECK-LABEL: func.func @canonical_zero_initial
// CHECK: vector.reduction
// CHECK-NOT: ondsp.reduce_mac
func.func @canonical_zero_initial(%lhs: memref<16xf32>, %rhs: memref<16xf32>) -> f32 {
  %initial = arith.constant 0.0 : f32
  %result = ondsp.reduce_mac %initial, %lhs, %rhs {
    numeric = #ondsp.fp<format = f32, contract = fast>
  } : (f32, memref<16xf32>, memref<16xf32>) -> f32
  return %result : f32
}
