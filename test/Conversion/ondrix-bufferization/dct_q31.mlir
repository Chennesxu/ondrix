// RUN: ondrix-opt %s --one-shot-bufferize="bufferize-function-boundaries function-boundary-type-conversion=identity-layout-map allow-return-allocs" | FileCheck %s
// RUN: ondrix-opt %s --one-shot-bufferize="bufferize-function-boundaries function-boundary-type-conversion=identity-layout-map allow-return-allocs" --canonicalize --vectorize-ondsp-fixed-decimate-outputs=vector-width=8 | FileCheck %s --check-prefix=LANES

// The Q31 profile reaches the same bufferized reduce_mac the Q15 DCT does,
// carrying the extent-derived narrowing as a requantized product against a
// constant row. Unlike Q15's saturating i40, the narrowed Q31 terms sum below
// 2^63, so the accumulator is the exact-modulo wrap class matmul's Q31 route
// takes and needs no prefix-range proof to reassociate.

// The Q31 tables are their own symbols: sharing the Q15 name would hand a
// reader of one profile the other profile's coefficients.
// CHECK-DAG: memref.global "private" constant @__ondrix_dct8_q31_row0 : memref<8xi32> = dense<2147483647>
// CHECK-DAG: memref.global "private" constant @__ondrix_dct8_q31_row4 : memref<8xi32> = dense<[1518500250, -1518500250, -1518500250, 1518500250, 1518500250, -1518500250, -1518500250, 1518500250]>

// CHECK-LABEL: func.func @dct8_q31(
// CHECK-SAME: %[[INPUT:.*]]: memref<8xi32>)
// CHECK-NOT: ondrix.dct
// CHECK: %[[OUTPUT:.*]] = memref.alloc() {{.*}} : memref<8xi32>
// CHECK: %[[ROW0:.*]] = memref.get_global @__ondrix_dct8_q31_row0 : memref<8xi32>
// CHECK: %[[INITIAL:.*]] = ondsp.acc_zero : <storage = i64, frac = 59, signed, update_overflow = wrap>
// CHECK: %[[REDUCED:.*]] = ondsp.reduce_mac %[[INITIAL]], %[[INPUT]], %[[ROW0]] {numeric = #ondsp.fixed<signed, storage = i32, frac = 31>, product = #ondsp.product<full, shift = 3, rounding = nearest_even>}
// Accumulator frac 62 - p read down to the declared output frac 30 - log2(N)
// is one acc_export, the tensor lowering's single round_shift boundary; no
// separate arithmetic shift is left on the path.
// CHECK: %[[BIN0:.*]] = ondsp.acc_export %[[REDUCED]] {dst = #ondsp.fixed<signed, storage = i32, frac = 27>, overflow = #ondsp.overflow<saturate>, rounding = #ondsp.rounding<nearest_even>} : (!ondsp.acc<storage = i64, frac = 59, signed, update_overflow = wrap>) -> i32
// CHECK-NOT: ondsp.round_shift
// CHECK: memref.store %[[BIN0]], %[[OUTPUT]][%{{.*}}]
// CHECK: memref.get_global @__ondrix_dct8_q31_row7 : memref<8xi32>
// CHECK: return %[[OUTPUT]]

func.func @dct8_q31(%input: tensor<8xi32>) -> tensor<8xi32> {
  %result = ondrix.dct %input {
    input_numeric = #ondsp.fixed<signed, storage = i32, frac = 31>,
    output_numeric = #ondsp.fixed<signed, storage = i32, frac = 27>,
    product_rounding = #ondsp.rounding<nearest_even>,
    rounding = #ondsp.rounding<nearest_even>
  } : (tensor<8xi32>) -> tensor<8xi32>
  return %result : tensor<8xi32>
}

// The product shift tracks the extent, so a wider DCT narrows harder and its
// accumulator sits one frac lower per doubling; both readings still export at
// a shift of 32.
// CHECK-LABEL: func.func @dct16_q31(
// CHECK: ondsp.acc_zero : <storage = i64, frac = 58, signed, update_overflow = wrap>
// CHECK: ondsp.reduce_mac {{.*}} product = #ondsp.product<full, shift = 4, rounding = nearest_even>}
// CHECK: ondsp.acc_export {{.*}} {dst = #ondsp.fixed<signed, storage = i32, frac = 26>

func.func @dct16_q31(%input: tensor<16xi32>) -> tensor<16xi32> {
  %result = ondrix.dct %input {
    input_numeric = #ondsp.fixed<signed, storage = i32, frac = 31>,
    output_numeric = #ondsp.fixed<signed, storage = i32, frac = 26>,
    product_rounding = #ondsp.rounding<nearest_even>,
    rounding = #ondsp.rounding<nearest_even>
  } : (tensor<16xi32>) -> tensor<16xi32>
  return %result : tensor<16xi32>
}

// Output batching claims the per-row reductions at the lane width, which is
// what the Q15 route needs its prefix-range proof for and this one does not.
// LANES-LABEL: func.func @dct8_q31
// LANES: ondsp.mac {{.*}} product = #ondsp.product<full, shift = 3, rounding = nearest_even>} : (!ondsp.acc<storage = i64, frac = 59, signed, update_overflow = wrap, lanes = 8>
// LANES: ondsp.acc_export {{.*}} -> vector<8xi32>
// LANES-NOT: ondsp.reduce_mac

// The raw-high profile: the row reduce_mac carries the high_raw product into
// the shared i40/frac30 wrapping state (at most 64 floors, the 2^36 bound),
// and the same acc_export reads it down by m onto the declared frac.
// CHECK-LABEL: func.func @dct8_q31_raw_high(
// CHECK: ondsp.acc_zero : <storage = i40, frac = 30, signed, update_overflow = wrap>
// CHECK: ondsp.reduce_mac {{.*}}product = #ondsp.product<high_raw>
// CHECK: ondsp.acc_export {{.*}}dst = #ondsp.fixed<signed, storage = i32, frac = 27>{{.*}} : (!ondsp.acc<storage = i40, frac = 30, signed, update_overflow = wrap>) -> i32
func.func @dct8_q31_raw_high(%input: tensor<8xi32>) -> tensor<8xi32> {
  %result = ondrix.dct %input {
    input_numeric = #ondsp.fixed<signed, storage = i32, frac = 31>,
    output_numeric = #ondsp.fixed<signed, storage = i32, frac = 27>,
    product = #ondsp.product<high_raw>
  } : (tensor<8xi32>) -> tensor<8xi32>
  return %result : tensor<8xi32>
}
