// RUN: ondrix-opt %s --vectorize-ondsp-fp-fast-memref-reduce | FileCheck %s
// RUN: not ondrix-opt %s --vectorize-ondsp-fp-fast-memref-reduce="vector-width=1" 2>&1 | FileCheck %s --check-prefix=WIDTH
// RUN: not ondrix-opt %s --vectorize-ondsp-fp-fast-memref-reduce="vector-width=5000" 2>&1 | FileCheck %s --check-prefix=WIDE

// The one transform that exploits the declared fast relaxation: W-lane fused
// partial sums, one cross-lane reduction, a scalar tail, every FP operation
// carrying exactly fastmath<reassoc,contract>. The pass description carries
// the authorization argument; off and fma sites below stay untouched.

// WIDTH: vector-width must be greater than one
// WIDE: vector-width must not exceed 4096

// CHECK-LABEL: func.func @f32_dot_fast_dynamic
// CHECK: cf.assert {{.*}}equal operand lengths
// CHECK: %[[PARTIAL:.*]] = scf.for {{.*}} iter_args(%[[ACC:.*]] = %{{.*}}) -> (vector<8xf32>)
// CHECK: vector.load {{.*}} : memref<?xf32>, vector<8xf32>
// CHECK: vector.load {{.*}} : memref<?xf32>, vector<8xf32>
// CHECK: math.fma {{.*}}, %[[ACC]] fastmath<reassoc,contract> : vector<8xf32>
// CHECK: %[[REDUCED:.*]] = vector.reduction <add>, %[[PARTIAL]] : vector<8xf32> into f32
// CHECK: %[[TAIL:.*]] = scf.for {{.*}} iter_args({{.*}} = %[[REDUCED]]) -> (f32)
// CHECK: math.fma {{.*}} fastmath<reassoc,contract> : f32
// CHECK: arith.addf %[[TAIL]], %{{.*}} fastmath<reassoc,contract> : f32
func.func @f32_dot_fast_dynamic(%lhs: memref<?xf32>, %rhs: memref<?xf32>) -> f32 {
  %zero = arith.constant 0.0 : f32
  %r = ondsp.reduce_mac %zero, %lhs, %rhs {numeric = #ondsp.fp<format = f32, contract = fast>} : (f32, memref<?xf32>, memref<?xf32>) -> f32
  return %r : f32
}

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
