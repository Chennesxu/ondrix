// RUN: ondrix-opt %s --one-shot-bufferize="bufferize-function-boundaries function-boundary-type-conversion=identity-layout-map allow-return-allocs" | FileCheck %s
// RUN: ondrix-opt %s --one-shot-bufferize="bufferize-function-boundaries function-boundary-type-conversion=identity-layout-map allow-return-allocs" --canonicalize --vectorize-ondsp-fixed-memref-reduce="vector-width=4" --parallelize-ondsp-fixed-wrap-vector-reduce --normalize-ondsp-fixed-vector-reduce | FileCheck %s --check-prefix=FULL-VECTOR

// The bufferized form squares the input with one reduction whose two operands
// are the same buffer. Squares are at most 2^30 and N is at most 4096, so the
// exact sum is at most 2^42: the i64 wrapping accumulator never wraps. An i40
// accumulator would not fit that range, which is why this reduction needs the
// wider wrapping accumulator admitted by the horizontal-domain predicate.
// Exporting to frac 30 - log2(N) makes acc_export divide by exactly 2^log2(N),
// which is the nearest-even saturating mean boundary of the tensor lowering.

// CHECK-LABEL: func.func @rms64_q15(
// CHECK-SAME: %[[INPUT:.*]]: memref<64xi16>)
// CHECK-NOT: ondrix.rms
// CHECK: %[[OUTPUT:.*]] = memref.alloc() {{.*}} : memref<1xi16>
// CHECK: %[[INITIAL:.*]] = ondsp.acc_zero : <storage = i64, frac = 30, signed, update_overflow = wrap>
// CHECK: %[[REDUCED:.*]] = ondsp.reduce_mac %[[INITIAL]], %[[INPUT]], %[[INPUT]] {numeric = #ondsp.fixed<signed, storage = i16, frac = 15>, product = #ondsp.product<full>}
// CHECK: %[[MEAN:.*]] = ondsp.acc_export %[[REDUCED]] {dst = #ondsp.fixed<signed, storage = i32, frac = 24>, overflow = #ondsp.overflow<saturate>, rounding = #ondsp.rounding<nearest_even>} : (!ondsp.acc<storage = i64, frac = 30, signed, update_overflow = wrap>) -> i32
// CHECK: %[[WIDE:.*]] = arith.extsi %[[MEAN]] : i32 to i64
// CHECK: %[[ROOT:.*]] = ondsp.sqrt_fixed %[[WIDE]] {rounding = #ondsp.rounding<nearest_even>} : (i64) -> i16
// CHECK: memref.store %[[ROOT]], %[[OUTPUT]]
// CHECK: return %[[OUTPUT]]

func.func @rms64_q15(%input: tensor<64xi16>) -> tensor<1xi16> {
  %result = ondrix.rms %input {
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    rounding = #ondsp.rounding<nearest_even>
  } : (tensor<64xi16>) -> tensor<1xi16>
  return %result : tensor<1xi16>
}

// The op's rounding attribute routes to the root, never to the mean: the mean
// stays nearest-even while the root takes the declared floor mode.
// CHECK-LABEL: func.func @rms64_floor_q15(
// CHECK: ondsp.acc_export {{.*}} {dst = #ondsp.fixed<signed, storage = i32, frac = 24>, overflow = #ondsp.overflow<saturate>, rounding = #ondsp.rounding<nearest_even>}
// CHECK: ondsp.sqrt_fixed {{.*}} {rounding = #ondsp.rounding<toward_negative>} : (i64) -> i16

func.func @rms64_floor_q15(%input: tensor<64xi16>) -> tensor<1xi16> {
  %result = ondrix.rms %input {
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    rounding = #ondsp.rounding<toward_negative>
  } : (tensor<64xi16>) -> tensor<1xi16>
  return %result : tensor<1xi16>
}

// The mean shift tracks the extent: N = 2 exports at frac 29.
// CHECK-LABEL: func.func @rms2_q15(
// CHECK: ondsp.acc_export {{.*}} {dst = #ondsp.fixed<signed, storage = i32, frac = 29>, overflow = #ondsp.overflow<saturate>, rounding = #ondsp.rounding<nearest_even>}

func.func @rms2_q15(%input: tensor<2xi16>) -> tensor<1xi16> {
  %result = ondrix.rms %input {
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    rounding = #ondsp.rounding<nearest_even>
  } : (tensor<2xi16>) -> tensor<1xi16>
  return %result : tensor<1xi16>
}

// The wrap horizontal path fires for the i64 wrapping accumulator.
// FULL-VECTOR-LABEL: func.func @rms64_q15
// FULL-VECTOR-COUNT-2: vector.load {{.*}} vector<4xi16>
// FULL-VECTOR: arith.muli {{.*}} : vector<4xi32>
// FULL-VECTOR: vector.reduction <add>, {{.*}} : vector<4xi64> into i64
// FULL-VECTOR: ondsp.acc_add_term {{.*}} : (!ondsp.acc<storage = i64, frac = 30, signed, update_overflow = wrap>, i64) -> !ondsp.acc<storage = i64, frac = 30, signed, update_overflow = wrap>
// FULL-VECTOR: ondsp.acc_export
// FULL-VECTOR: ondsp.sqrt_fixed
// FULL-VECTOR-LABEL: func.func @rms64_floor_q15
// FULL-VECTOR: vector.reduction <add>, {{.*}} : vector<4xi64> into i64
// FULL-VECTOR-LABEL: func.func @rms2_q15
// FULL-VECTOR-NOT: ondsp.reduce_mac
