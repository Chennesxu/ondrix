// RUN: ondrix-opt %s --vectorize-ondsp-fp-fast-memref-reduce=vector-width=8 | FileCheck %s

// The initial value is the fold's accumulator, so it enters the rebuilt tree
// exactly once and nothing synthesizes a start value. A literal +0.0 anywhere
// — lane seed or fold start — is a term the source graph does not have, and it
// turns an all-negative-zero reduction's declared -0.0 into +0.0.

// CHECK-LABEL: func.func @negative_zero_initial
// CHECK: %[[INIT:.*]] = arith.constant -0.000000e+00 : f32
// CHECK: vector.reduction <add>, %{{[^ ]*}}, %[[INIT]] :
// CHECK-NOT: ondsp.reduce_mac
func.func @negative_zero_initial(%lhs: memref<16xf32>, %rhs: memref<16xf32>) -> f32 {
  %initial = arith.constant -0.0 : f32
  %result = ondsp.reduce_mac %initial, %lhs, %rhs {
    numeric = #ondsp.fp<format = f32, contract = fast>
  } : (f32, memref<16xf32>, memref<16xf32>) -> f32
  return %result : f32
}

// CHECK-LABEL: func.func @dynamic_initial
// CHECK: vector.reduction <add>, %{{[^ ]*}}, %arg0 :
// CHECK-NOT: ondsp.reduce_mac
func.func @dynamic_initial(%initial: f32, %lhs: memref<16xf32>, %rhs: memref<16xf32>) -> f32 {
  %result = ondsp.reduce_mac %initial, %lhs, %rhs {
    numeric = #ondsp.fp<format = f32, contract = fast>
  } : (f32, memref<16xf32>, memref<16xf32>) -> f32
  return %result : f32
}

// Below the lane count there is no lane to fill, so a statically short
// reduction is left for the ordered scalar lowering rather than padded.
// CHECK-LABEL: func.func @single_term
// CHECK-NOT: vector.load
// CHECK: ondsp.reduce_mac
func.func @single_term(%initial: f32, %lhs: memref<1xf32>, %rhs: memref<1xf32>) -> f32 {
  %result = ondsp.reduce_mac %initial, %lhs, %rhs {
    numeric = #ondsp.fp<format = f32, contract = fast>
  } : (f32, memref<1xf32>, memref<1xf32>) -> f32
  return %result : f32
}

// CHECK-LABEL: func.func @empty_reduction
// CHECK-NOT: vector.load
// CHECK: ondsp.reduce_mac
func.func @empty_reduction(%initial: f32, %lhs: memref<0xf32>, %rhs: memref<0xf32>) -> f32 {
  %result = ondsp.reduce_mac %initial, %lhs, %rhs {
    numeric = #ondsp.fp<format = f32, contract = fast>
  } : (f32, memref<0xf32>, memref<0xf32>) -> f32
  return %result : f32
}
