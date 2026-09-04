// RUN: ondrix-opt %s --one-shot-bufferize="bufferize-function-boundaries function-boundary-type-conversion=identity-layout-map allow-return-allocs" | FileCheck %s
// RUN: ondrix-opt %s --one-shot-bufferize="bufferize-function-boundaries function-boundary-type-conversion=identity-layout-map allow-return-allocs" --canonicalize --vectorize-ondsp-fixed-decimate-outputs=vector-width=8 | FileCheck %s --check-prefix=LANES

// The Q31 profile reaches the same bufferized reduce_mac as Q15, carrying the
// extent-derived narrowing as a requantized product. The i64 wrapping
// accumulator never wraps: K terms bounded by 2^(62 - p) with p = floor(log2 K)
// sum below 2^63, so it stays the exact-modulo reassociation class.

// CHECK-LABEL: func.func @matmul8x8x8_q31(
// CHECK-SAME: %[[LHS:.*]]: memref<8x8xi32>, %[[RHS:.*]]: memref<8x8xi32>)
// CHECK-NOT: ondrix.matmul
// CHECK: %[[OUTPUT:.*]] = memref.alloc() {{.*}} : memref<8x8xi32>
// CHECK: %[[PACKED:.*]] = memref.alloc() {{.*}} : memref<8x8xi32>
// CHECK: %[[LHS_ROW:.*]] = memref.subview %[[LHS]][%{{.*}}, 0] [1, 8] [1, 1] : memref<8x8xi32> to memref<8xi32, strided<[1], offset: ?>>
// CHECK: %[[PACKED_ROW:.*]] = memref.subview %[[PACKED]][%{{.*}}, 0] [1, 8] [1, 1] : memref<8x8xi32> to memref<8xi32, strided<[1], offset: ?>>
// CHECK: %[[INITIAL:.*]] = ondsp.acc_zero : <storage = i64, frac = 59, signed, update_overflow = wrap>
// CHECK: %[[REDUCED:.*]] = ondsp.reduce_mac %[[INITIAL]], %[[LHS_ROW]], %[[PACKED_ROW]] {numeric = #ondsp.fixed<signed, storage = i32, frac = 31>, product = #ondsp.product<full, shift = 3, rounding = nearest_even>}
// CHECK: %[[EXPORTED:.*]] = ondsp.acc_export %[[REDUCED]] {dst = #ondsp.fixed<signed, storage = i32, frac = 31>, overflow = #ondsp.overflow<saturate>, rounding = #ondsp.rounding<nearest_even>} : (!ondsp.acc<storage = i64, frac = 59, signed, update_overflow = wrap>) -> i32
// CHECK: memref.store %[[EXPORTED]], %[[OUTPUT]]
// CHECK: memref.dealloc %[[PACKED]]

func.func @matmul8x8x8_q31(%a: tensor<8x8xi32>, %b: tensor<8x8xi32>) -> tensor<8x8xi32> {
  %c = ondrix.matmul %a, %b {
    numeric = #ondsp.fixed<signed, storage = i32, frac = 31>,
    product_rounding = #ondsp.rounding<nearest_even>,
    rounding = #ondsp.rounding<nearest_even>
  } : (tensor<8x8xi32>, tensor<8x8xi32>) -> tensor<8x8xi32>
  return %c : tensor<8x8xi32>
}

// K = 1 has no product boundary at all, so the accumulator stays at frac 62
// and the operation declares no product_rounding.
// CHECK-LABEL: func.func @matmul1x1x1_q31(
// CHECK: ondsp.acc_zero : <storage = i64, frac = 62, signed, update_overflow = wrap>
// CHECK: ondsp.reduce_mac {{.*}} product = #ondsp.product<full>}
// CHECK: ondsp.acc_export {{.*}} (!ondsp.acc<storage = i64, frac = 62, signed, update_overflow = wrap>) -> i32

func.func @matmul1x1x1_q31(%a: tensor<1x1xi32>, %b: tensor<1x1xi32>) -> tensor<1x1xi32> {
  %c = ondrix.matmul %a, %b {
    numeric = #ondsp.fixed<signed, storage = i32, frac = 31>,
    rounding = #ondsp.rounding<nearest_even>
  } : (tensor<1x1xi32>, tensor<1x1xi32>) -> tensor<1x1xi32>
  return %c : tensor<1x1xi32>
}

// Output batching claims the whole column loop, packed copy included, and the
// requantized product travels into the multi-lane accumulator unchanged.
// LANES-LABEL: func.func @matmul8x8x8_q31
// LANES: vector.load {{.*}} : memref<8x8xi32>, vector<8xi32>
// LANES: ondsp.mac {{.*}} product = #ondsp.product<full, shift = 3, rounding = nearest_even>} : (!ondsp.acc<storage = i64, frac = 59, signed, update_overflow = wrap, lanes = 8>, vector<8xi32>, i32)
// LANES: ondsp.acc_export {{.*}} -> vector<8xi32>
// LANES: vector.store
// LANES-NOT: ondsp.reduce_mac
// A K = 1 reduction has no full block to batch and keeps the ordered form.
// LANES-LABEL: func.func @matmul1x1x1_q31
// LANES: ondsp.reduce_mac
