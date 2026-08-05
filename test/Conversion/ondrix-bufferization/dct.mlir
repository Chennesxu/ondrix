// RUN: ondrix-opt %s --one-shot-bufferize="bufferize-function-boundaries function-boundary-type-conversion=identity-layout-map allow-return-allocs" | FileCheck %s
// RUN: ondrix-opt %s --one-shot-bufferize="bufferize-function-boundaries function-boundary-type-conversion=identity-layout-map allow-return-allocs" --canonicalize --vectorize-ondsp-constant-saturating-memref-reduce="vector-width=4 max-elements=64 proof-trace-output=%t.proof.json" | FileCheck %s --check-prefix=FULL-VECTOR --implicit-check-not=ondsp.reduce_mac
// RUN: FileCheck %s --check-prefix=TRACE --input-file=%t.proof.json
// RUN: ondrix-opt %s --one-shot-bufferize="bufferize-function-boundaries function-boundary-type-conversion=identity-layout-map allow-return-allocs" | FileCheck %s --check-prefix=ORDER
// RUN: ondrix-opt %s --one-shot-bufferize="bufferize-function-boundaries allow-return-allocs" --canonicalize --vectorize-ondsp-constant-saturating-memref-reduce="vector-width=4 max-elements=64" | FileCheck %s --check-prefix=DYNAMIC-LAYOUT --implicit-check-not=vector.load --implicit-check-not=vector.reduction

// The bufferized form is a second consumer of the same DCT contract: one
// zero-seeded reduction and one nearest-even saturating boundary per output
// row. Unlike matmul/rms (runtime values, wrapping accumulators, proof-free
// exact-modulo reassociation), the constant DCT row declares a SATURATING
// accumulator, so the horizontal Vector consumer must earn its reassociation
// through the constant-coefficient prefix-range proof — the derivation lives
// in dct_q15_vector_aot.mlir.

// The tie-guarded coefficient tables are materialized as immutable rank-1
// globals, one per output row, because that is the only constant form
// `ConstantSequenceAnalysis` resolves: its view walk accepts rank-1 subviews
// and casts over a `memref.get_global` and refuses any rank-2 source.
// CHECK-DAG: memref.global "private" constant @__ondrix_dct8_row0 : memref<8xi16> = dense<32767>
// CHECK-DAG: memref.global "private" constant @__ondrix_dct8_row1 : memref<8xi16> = dense<[32138, 27246, 18205, 6393, -6393, -18205, -27246, -32138]>
// CHECK-DAG: memref.global "private" constant @__ondrix_dct8_row4 : memref<8xi16> = dense<[23170, -23170, -23170, 23170, 23170, -23170, -23170, 23170]>
// CHECK-DAG: memref.global "private" constant @__ondrix_dct8_row7 : memref<8xi16> = dense<[6393, -18205, 27246, -32138, 32138, -27246, 18205, -6393]>
// CHECK-DAG: memref.global "private" constant @__ondrix_dct4_row1 : memref<4xi16> = dense<[30274, 12540, -12540, -30274]>

// The reserved globals appear in symbol order whatever order the functions
// bufferize in; emission-order placement fails these ordered checks.
// ORDER: memref.global "private" constant @__ondrix_dct4_row1
// ORDER: memref.global "private" constant @__ondrix_dct8_row0
// ORDER: memref.global "private" constant @__ondrix_dct8_row1
// ORDER: memref.global "private" constant @__ondrix_dct8_row4
// ORDER: memref.global "private" constant @__ondrix_dct8_row7

// CHECK-LABEL: func.func @dct8_q15(
// CHECK-SAME: %[[INPUT:.*]]: memref<8xi16>)
// CHECK-NOT: ondrix.dct
// CHECK: %[[OUTPUT:.*]] = memref.alloc() {{.*}} : memref<8xi16>
// CHECK: %[[ROW0:.*]] = memref.get_global @__ondrix_dct8_row0 : memref<8xi16>
// CHECK: %[[INITIAL:.*]] = ondsp.acc_zero : <storage = i40, frac = 30, signed, update_overflow = saturate>
// CHECK: %[[REDUCED:.*]] = ondsp.reduce_mac %[[INITIAL]], %[[INPUT]], %[[ROW0]] {numeric = #ondsp.fixed<signed, storage = i16, frac = 15>, product = #ondsp.product<full>}
// The identity i64/frac30 export is a WIDENING acc_export from i40: the raw
// sum keeps its declared reading and the value-changing scaling stays in the
// arithmetic round_shift below.
// CHECK: %[[RAW:.*]] = ondsp.acc_export %[[REDUCED]] {dst = #ondsp.fixed<signed, storage = i64, frac = 30>, overflow = #ondsp.overflow<saturate>, rounding = #ondsp.rounding<nearest_even>} : (!ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>) -> i64
// CHECK: %[[BIN0:.*]] = ondsp.round_shift %[[RAW]] {scale = #ondsp.scale<pre_shift_left = 0, post_shift_right = 19, rounding = nearest_even, overflow = saturate, saturate_to = i16>} : (i64) -> i16
// CHECK: memref.store %[[BIN0]], %[[OUTPUT]][%{{.*}}]
// CHECK: %[[ROW1:.*]] = memref.get_global @__ondrix_dct8_row1 : memref<8xi16>
// CHECK: ondsp.reduce_mac %{{.*}}, %[[INPUT]], %[[ROW1]]
// CHECK: memref.get_global @__ondrix_dct8_row7 : memref<8xi16>
// CHECK: return %[[OUTPUT]]

func.func @dct8_q15(%input: tensor<8xi16>) -> tensor<8xi16> {
  %result = ondrix.dct %input {
    input_numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    output_numeric = #ondsp.fixed<signed, storage = i16, frac = 11>
  } : (tensor<8xi16>) -> tensor<8xi16>
  return %result : tensor<8xi16>
}

// A second operation of the same extent reuses the existing tables instead of
// emitting a colliding second definition; a duplicate symbol would fail the
// module verifier rather than reach these checks.
// CHECK-LABEL: func.func @dct8_shared_tables_q15(
// CHECK: memref.get_global @__ondrix_dct8_row0 : memref<8xi16>
// CHECK: ondsp.reduce_mac

func.func @dct8_shared_tables_q15(%input: tensor<8xi16>) -> tensor<8xi16> {
  %result = ondrix.dct %input {
    input_numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    output_numeric = #ondsp.fixed<signed, storage = i16, frac = 11>
  } : (tensor<8xi16>) -> tensor<8xi16>
  return %result : tensor<8xi16>
}

// The boundary shift tracks the extent exactly as in the tensor lowering:
// 16 + log2(N) is 18 at N = 4, onto the frac = 12 unnormalized reading.
// CHECK-LABEL: func.func @dct4_q15(
// CHECK: memref.get_global @__ondrix_dct4_row0 : memref<4xi16>
// CHECK: ondsp.acc_zero : <storage = i40, frac = 30, signed, update_overflow = saturate>
// CHECK: ondsp.round_shift %{{.*}} {scale = #ondsp.scale<pre_shift_left = 0, post_shift_right = 18, rounding = nearest_even, overflow = saturate, saturate_to = i16>} : (i64) -> i16

func.func @dct4_q15(%input: tensor<4xi16>) -> tensor<4xi16> {
  %result = ondrix.dct %input {
    input_numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    output_numeric = #ondsp.fixed<signed, storage = i16, frac = 12>
  } : (tensor<4xi16>) -> tensor<4xi16>
  return %result : tensor<4xi16>
}

// The constant-coefficient prefix-range proof authorizes the horizontal form:
// every chunked prefix of a row stays inside the i40 accumulator because
// sum_n |c[k][n]| * 32768 <= 64 * 32767 * 32768 < 2^39, so no reassociated
// partial sum can saturate. The wrap shortcut is unavailable here by
// construction, which is exactly what this route demonstrates.
// FULL-VECTOR-LABEL: func.func @dct8_q15
// FULL-VECTOR-COUNT-2: vector.load {{.*}} vector<4xi16>
// FULL-VECTOR: arith.muli {{.*}} : vector<4xi32>
// FULL-VECTOR: vector.reduction <add>, {{.*}} : vector<4xi64> into i64
// FULL-VECTOR: ondsp.acc_add_term {{.*}} : (!ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>, i64) -> !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>
// FULL-VECTOR: ondsp.acc_export {{.*}} -> i64
// FULL-VECTOR: ondsp.round_shift
// FULL-VECTOR-LABEL: func.func @dct8_shared_tables_q15
// FULL-VECTOR: vector.reduction <add>, {{.*}} : vector<4xi64> into i64
// FULL-VECTOR-LABEL: func.func @dct4_q15
// FULL-VECTOR: vector.reduction <add>, {{.*}} : vector<4xi64> into i64

// Every DCT row is recorded as its own audit subject, with the saturating
// accumulator that made the proof necessary.
// TRACE-DAG: "vector_width":{{ *}}4
// TRACE-DAG: "accumulator_update_overflow":"saturate"
// TRACE-DAG: "accumulator_storage_width":{{ *}}40
// TRACE-DAG: "subject_ordinal":{{ *}}19

// The unit-stride precondition of vectorize-ondsp-fixed-memref-reduce: the
// coefficient tables stay unit-stride globals, but a dynamically laid out
// input buffer alone keeps the reduction an ordered scalar fallback.
// DYNAMIC-LAYOUT-LABEL: func.func @dct8_q15(
// DYNAMIC-LAYOUT-SAME: memref<8xi16, strided<[?], offset: ?>>
// DYNAMIC-LAYOUT: ondsp.reduce_mac
// DYNAMIC-LAYOUT: ondsp.acc_export {{.*}} -> i64
// DYNAMIC-LAYOUT: ondsp.round_shift
// DYNAMIC-LAYOUT-LABEL: func.func @dct8_shared_tables_q15(
// DYNAMIC-LAYOUT: ondsp.reduce_mac
// DYNAMIC-LAYOUT-LABEL: func.func @dct4_q15(
// DYNAMIC-LAYOUT: ondsp.reduce_mac
