// RUN: ondrix-opt %s --vectorize-ondsp-fp-fast-memref-reduce=vector-width=8 | FileCheck %s

// The initial value is carried, never reproduced: whatever it is, it enters
// the rebuilt tree exactly once. That is why these cases vectorize where an
// identity-seeded rewrite had to refuse them — with lanes seeded by a
// synthesized +0.0, an initial of -0.0 over one term gave +0.0 against the
// declared -0.0, and one term admits no reassociation to authorize it.

// CHECK-LABEL: func.func @negative_zero_initial
// CHECK: %[[INIT:.*]] = arith.constant -0.000000e+00 : f32
// CHECK: vector.reduction
// CHECK: arith.addf %[[INIT]], %{{.*}} : f32
// CHECK-NOT: ondsp.reduce_mac
func.func @negative_zero_initial(%lhs: memref<16xf32>, %rhs: memref<16xf32>) -> f32 {
  %initial = arith.constant -0.0 : f32
  %result = ondsp.reduce_mac %initial, %lhs, %rhs {
    numeric = #ondsp.fp<format = f32, contract = fast>
  } : (f32, memref<16xf32>, memref<16xf32>) -> f32
  return %result : f32
}

// CHECK-LABEL: func.func @dynamic_initial
// CHECK: vector.reduction
// CHECK: arith.addf %arg0, %{{.*}} : f32
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
