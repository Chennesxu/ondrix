// RUN: ondrix-opt %s --lower-ondsp-f32-reduce-to-scalar="vector-width=4" | FileCheck %s
// RUN: ondrix-opt %s --lower-ondsp-f32-reduce-to-scalar | FileCheck %s --check-prefix=SCALAR

// The products are batched and the four folds stay in lane order, so the event
// graph is the one the off contract declares and no permission is recorded.
func.func @off_batches_products(%lhs: memref<9xf32>, %rhs: memref<9xf32>, %seed: f32) -> f32 {
  %result = ondsp.reduce_mac %seed, %lhs, %rhs {
    numeric = #ondsp.fp<format = f32, contract = off>
  } : (f32, memref<9xf32>, memref<9xf32>) -> f32
  return %result : f32
}

// CHECK-LABEL: func.func @off_batches_products
// CHECK: vector.load {{.*}} : memref<9xf32>, vector<4xf32>
// CHECK: %[[P:.*]] = arith.mulf {{.*}} : vector<4xf32>
// CHECK: %[[L0:.*]] = vector.extract %[[P]][0]
// CHECK: %[[A0:.*]] = arith.addf %{{.*}}, %[[L0]] : f32
// CHECK: %[[L1:.*]] = vector.extract %[[P]][1]
// CHECK: %[[A1:.*]] = arith.addf %[[A0]], %[[L1]] : f32
// CHECK: %[[L2:.*]] = vector.extract %[[P]][2]
// CHECK: %[[A2:.*]] = arith.addf %[[A1]], %[[L2]] : f32
// CHECK: %[[L3:.*]] = vector.extract %[[P]][3]
// CHECK: arith.addf %[[A2]], %[[L3]] : f32
// CHECK-NOT: ondsp.fast_used
// Nine elements leave one ordered product behind the two full blocks.
// CHECK: scf.for
// CHECK: arith.mulf {{.*}} : f32

// SCALAR-LABEL: func.func @off_batches_products
// SCALAR-NOT: vector.load

// A fused contract declared the multiply and the add to be one event, so it has
// no separable product to batch.
func.func @fma_keeps_the_scalar_chain(%lhs: memref<8xf32>, %rhs: memref<8xf32>, %seed: f32) -> f32 {
  %result = ondsp.reduce_mac %seed, %lhs, %rhs {
    numeric = #ondsp.fp<format = f32, contract = fma>
  } : (f32, memref<8xf32>, memref<8xf32>) -> f32
  return %result : f32
}

// CHECK-LABEL: func.func @fma_keeps_the_scalar_chain
// CHECK-NOT: vector.load
// CHECK: math.fma

// A reversed coefficient view is not contiguous, so there is no vector load to
// make and the reduction keeps its ordered scalar form.
func.func @reversed_view_refuses(%lhs: memref<8xf32>,
    %rhs: memref<8xf32, strided<[-1], offset: 7>>, %seed: f32) -> f32 {
  %result = ondsp.reduce_mac %seed, %lhs, %rhs {
    numeric = #ondsp.fp<format = f32, contract = off>
  } : (f32, memref<8xf32>, memref<8xf32, strided<[-1], offset: 7>>) -> f32
  return %result : f32
}

// CHECK-LABEL: func.func @reversed_view_refuses
// CHECK-NOT: vector.load
// CHECK: arith.mulf {{.*}} : f32
