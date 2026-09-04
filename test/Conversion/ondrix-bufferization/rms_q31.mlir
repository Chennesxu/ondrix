// RUN: ondrix-opt %s --one-shot-bufferize="bufferize-function-boundaries function-boundary-type-conversion=identity-layout-map allow-return-allocs" | FileCheck %s
// RUN: ondrix-opt %s --one-shot-bufferize="bufferize-function-boundaries function-boundary-type-conversion=identity-layout-map allow-return-allocs" --canonicalize --vectorize-ondsp-fixed-elementwise-loops=vector-width=8 --vectorize-ondsp-fixed-memref-reduce="vector-width=8 chunk-multiple=4" --normalize-ondsp-fixed-vector-reduce | FileCheck %s --check-prefix=FULL-VECTOR --implicit-check-not=ondsp.reduce_mac

// The Q31 profile carries its extent-derived narrowing as a pre-requantized
// scratch copy, which the reduction then squares exactly: N squares of a
// Q1.(31 - k) value sum below 2^63, so the i64 wrapping accumulator never
// wraps and stays the exact-modulo reassociation class. The left shift by 2k
// restores the scale the copy dropped, before the root.

// CHECK-LABEL: func.func @rms4096_q31(
// CHECK-SAME: %[[INPUT:.*]]: memref<4096xi32>)
// CHECK-NOT: ondrix.rms
// CHECK: %[[OUTPUT:.*]] = memref.alloc() {{.*}} : memref<1xi32>
// CHECK: %[[SCRATCH:.*]] = memref.alloc() {{.*}} : memref<4096xi32>
// CHECK: scf.for %[[POSITION:.*]] =
// CHECK: %[[ELEMENT:.*]] = memref.load %[[INPUT]][%[[POSITION]]]
// CHECK: %[[SCALED:.*]] = ondsp.round_shift %[[ELEMENT]] {scale = #ondsp.scale<pre_shift_left = 0, post_shift_right = 6, rounding = nearest_even, overflow = saturate, saturate_to = i32>} : (i32) -> i32
// CHECK: memref.store %[[SCALED]], %[[SCRATCH]][%[[POSITION]]]
// CHECK: %[[INITIAL:.*]] = ondsp.acc_zero : <storage = i64, frac = 62, signed, update_overflow = wrap>
// CHECK: %[[REDUCED:.*]] = ondsp.reduce_mac %[[INITIAL]], %[[SCRATCH]], %[[SCRATCH]] {numeric = #ondsp.fixed<signed, storage = i32, frac = 31>, product = #ondsp.product<full>}
// CHECK: %[[SUM:.*]] = ondsp.acc_export %[[REDUCED]] {dst = #ondsp.fixed<signed, storage = i64, frac = 62>, overflow = #ondsp.overflow<saturate>, rounding = #ondsp.rounding<nearest_even>} : (!ondsp.acc<storage = i64, frac = 62, signed, update_overflow = wrap>) -> i64
// CHECK: %[[MEAN:.*]] = ondsp.round_shift %[[SUM]] {scale = #ondsp.scale<pre_shift_left = 0, post_shift_right = 12, rounding = nearest_even, overflow = saturate, saturate_to = i64>} : (i64) -> i64
// CHECK: %[[RESTORED:.*]] = arith.shli %[[MEAN]], %{{.*}} : i64
// CHECK: %[[ROOT:.*]] = ondsp.sqrt_fixed %[[RESTORED]] {rounding = #ondsp.rounding<nearest_even>} : (i64) -> i32
// CHECK: memref.store %[[ROOT]], %[[OUTPUT]]
// CHECK: memref.dealloc %[[SCRATCH]]

func.func @rms4096_q31(%input: tensor<4096xi32>) -> tensor<1xi32> {
  %result = ondrix.rms %input {
    numeric = #ondsp.fixed<signed, storage = i32, frac = 31>,
    input_rounding = #ondsp.rounding<nearest_even>,
    rounding = #ondsp.rounding<nearest_even>
  } : (tensor<4096xi32>) -> tensor<1xi32>
  return %result : tensor<1xi32>
}

// Both shifts track the extent independently: N = 2 pre-shifts by 1 and means
// by 2^1, where N = 4096 pre-shifts by 6 and means by 2^12. The scratch copy
// reads as Q31 either way, so the accumulator frac does not move.
// CHECK-LABEL: func.func @rms2_q31(
// CHECK: ondsp.round_shift {{.*}} {scale = #ondsp.scale<pre_shift_left = 0, post_shift_right = 1, rounding = nearest_even, overflow = saturate, saturate_to = i32>} : (i32) -> i32
// CHECK: ondsp.acc_zero : <storage = i64, frac = 62, signed, update_overflow = wrap>
// CHECK: ondsp.round_shift {{.*}} {scale = #ondsp.scale<pre_shift_left = 0, post_shift_right = 1, rounding = nearest_even, overflow = saturate, saturate_to = i64>} : (i64) -> i64
// CHECK: %[[SHORT_RESTORE:.*]] = arith.constant 2 : i64
// CHECK: arith.shli %{{.*}}, %[[SHORT_RESTORE]] : i64
// CHECK: ondsp.sqrt_fixed

func.func @rms2_q31(%input: tensor<2xi32>) -> tensor<1xi32> {
  %result = ondrix.rms %input {
    numeric = #ondsp.fixed<signed, storage = i32, frac = 31>,
    input_rounding = #ondsp.rounding<nearest_even>,
    rounding = #ondsp.rounding<nearest_even>
  } : (tensor<2xi32>) -> tensor<1xi32>
  return %result : tensor<1xi32>
}

// Both stages take lanes: the pre-requantization as an elementwise map and the
// sum of squares as the wrap horizontal reduction.
// FULL-VECTOR-LABEL: func.func @rms4096_q31
// FULL-VECTOR: vector.load {{.*}} : memref<4096xi32>, vector<8xi32>
// FULL-VECTOR: ondsp.round_shift {{.*}} : (vector<8xi32>) -> vector<8xi32>
// FULL-VECTOR: vector.store {{.*}} : memref<4096xi32>, vector<8xi32>
// FULL-VECTOR: arith.muli {{.*}} : vector<8xi64>
// FULL-VECTOR: vector.reduction <add>, {{.*}} : vector<8xi64> into i64
// FULL-VECTOR: ondsp.acc_export
// FULL-VECTOR: ondsp.sqrt_fixed
