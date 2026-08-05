// RUN: ondrix-opt %s --one-shot-bufferize="bufferize-function-boundaries function-boundary-type-conversion=identity-layout-map allow-return-allocs" | FileCheck %s
// RUN: ondrix-opt %s --one-shot-bufferize="bufferize-function-boundaries function-boundary-type-conversion=identity-layout-map allow-return-allocs" --canonicalize --vectorize-ondsp-fixed-memref-reduce="vector-width=4" --parallelize-ondsp-fixed-wrap-vector-reduce --normalize-ondsp-fixed-vector-reduce | FileCheck %s --check-prefix=FULL-VECTOR --implicit-check-not=ondsp.reduce_mac
// RUN: ondrix-opt %s --one-shot-bufferize="bufferize-function-boundaries allow-return-allocs" --canonicalize --vectorize-ondsp-fixed-memref-reduce="vector-width=4" --parallelize-ondsp-fixed-wrap-vector-reduce --normalize-ondsp-fixed-vector-reduce | FileCheck %s --check-prefix=DYNAMIC-LAYOUT --implicit-check-not=vector.load --implicit-check-not=vector.reduction

// The bufferized form is a second consumer of the same matmul contract: one
// zero-seeded ordered reduction and one nearest-even saturating boundary per
// output element. The i40 wrapping accumulator never wraps (K <= 64 products
// of magnitude at most 2^30), so it is the exact-modulo reassociation class
// and the horizontal Vector consumer needs no prefix proof.

// CHECK-LABEL: func.func @matmul8x8x8_q15(
// CHECK-SAME: %[[LHS:.*]]: memref<8x8xi16>, %[[RHS:.*]]: memref<8x8xi16>)
// CHECK-NOT: ondrix.matmul
// CHECK: %[[OUTPUT:.*]] = memref.alloc() {{.*}} : memref<8x8xi16>
// The transposed pack turns stride-N columns of B into unit-stride rows.
// CHECK: %[[PACKED:.*]] = memref.alloc() {{.*}} : memref<8x8xi16>
// CHECK: scf.for %[[PACK_COLUMN:.*]] =
// CHECK: scf.for %[[PACK_INDEX:.*]] =
// CHECK: %[[ELEMENT:.*]] = memref.load %[[RHS]][%[[PACK_INDEX]], %[[PACK_COLUMN]]]
// CHECK: memref.store %[[ELEMENT]], %[[PACKED]][%[[PACK_COLUMN]], %[[PACK_INDEX]]]
// CHECK: scf.for %[[ROW:.*]] =
// CHECK: %[[LHS_ROW:.*]] = memref.subview %[[LHS]][%[[ROW]], 0] [1, 8] [1, 1] : memref<8x8xi16> to memref<8xi16, strided<[1], offset: ?>>
// CHECK: scf.for %[[COLUMN:.*]] =
// CHECK: %[[PACKED_ROW:.*]] = memref.subview %[[PACKED]][%[[COLUMN]], 0] [1, 8] [1, 1] : memref<8x8xi16> to memref<8xi16, strided<[1], offset: ?>>
// CHECK: %[[INITIAL:.*]] = ondsp.acc_zero : <storage = i40, frac = 30, signed, update_overflow = wrap>
// CHECK: %[[REDUCED:.*]] = ondsp.reduce_mac %[[INITIAL]], %[[LHS_ROW]], %[[PACKED_ROW]] {numeric = #ondsp.fixed<signed, storage = i16, frac = 15>, product = #ondsp.product<full>}
// CHECK: %[[EXPORTED:.*]] = ondsp.acc_export %[[REDUCED]] {dst = #ondsp.fixed<signed, storage = i16, frac = 15>, overflow = #ondsp.overflow<saturate>, rounding = #ondsp.rounding<nearest_even>} : (!ondsp.acc<storage = i40, frac = 30, signed, update_overflow = wrap>) -> i16
// CHECK: memref.store %[[EXPORTED]], %[[OUTPUT]][%[[ROW]], %[[COLUMN]]]
// CHECK: memref.dealloc %[[PACKED]]
// CHECK: return %[[OUTPUT]]

func.func @matmul8x8x8_q15(%a: tensor<8x8xi16>, %b: tensor<8x8xi16>) -> tensor<8x8xi16> {
  %c = ondrix.matmul %a, %b {
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    rounding = #ondsp.rounding<nearest_even>
  } : (tensor<8x8xi16>, tensor<8x8xi16>) -> tensor<8x8xi16>
  return %c : tensor<8x8xi16>
}

// All three dimensions stay independent: the scratch buffer is N x K and the
// reduction length is K.
// CHECK-LABEL: func.func @matmul4x16x3_q15(
// CHECK: memref.alloc() {{.*}} : memref<4x3xi16>
// CHECK: %[[RECT_PACKED:.*]] = memref.alloc() {{.*}} : memref<3x16xi16>
// CHECK: memref.subview %{{.*}}[%{{.*}}, 0] [1, 16] [1, 1] : memref<4x16xi16> to memref<16xi16, strided<[1], offset: ?>>
// CHECK: memref.subview %[[RECT_PACKED]][%{{.*}}, 0] [1, 16] [1, 1] : memref<3x16xi16> to memref<16xi16, strided<[1], offset: ?>>
// CHECK: ondsp.reduce_mac
// CHECK: ondsp.acc_export

func.func @matmul4x16x3_q15(%a: tensor<4x16xi16>, %b: tensor<16x3xi16>) -> tensor<4x3xi16> {
  %c = ondrix.matmul %a, %b {
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    rounding = #ondsp.rounding<nearest_even>
  } : (tensor<4x16xi16>, tensor<16x3xi16>) -> tensor<4x3xi16>
  return %c : tensor<4x3xi16>
}

// Both unit-stride operands reach the existing wrap horizontal reduction.
// FULL-VECTOR-LABEL: func.func @matmul8x8x8_q15
// FULL-VECTOR-COUNT-2: vector.load {{.*}} vector<4xi16>
// FULL-VECTOR: arith.muli {{.*}} : vector<4xi32>
// FULL-VECTOR: vector.reduction <add>, {{.*}} : vector<4xi64> into i64
// FULL-VECTOR: ondsp.acc_add_term
// FULL-VECTOR: ondsp.acc_export
// FULL-VECTOR-LABEL: func.func @matmul4x16x3_q15
// FULL-VECTOR: vector.reduction <add>, {{.*}} : vector<4xi64> into i64

// The unit-stride precondition of vectorize-ondsp-fixed-memref-reduce: the
// A-row view inherits the dynamic layout of the incoming buffer, so the
// reduction stays an ordered scalar fallback.
// DYNAMIC-LAYOUT-LABEL: func.func @matmul8x8x8_q15(
// DYNAMIC-LAYOUT-SAME: memref<8x8xi16, strided<[?, ?], offset: ?>>
// DYNAMIC-LAYOUT: memref.subview %{{.*}}[%{{.*}}, 0] [1, 8] [1, 1] : memref<8x8xi16, strided<[?, ?], offset: ?>> to memref<8xi16, strided<[?], offset: ?>>
// DYNAMIC-LAYOUT: ondsp.reduce_mac
// DYNAMIC-LAYOUT: ondsp.acc_export
// DYNAMIC-LAYOUT-LABEL: func.func @matmul4x16x3_q15(
// DYNAMIC-LAYOUT: ondsp.reduce_mac
// DYNAMIC-LAYOUT-NOT: vector.reduction
// DYNAMIC-LAYOUT-NOT: vector.load
