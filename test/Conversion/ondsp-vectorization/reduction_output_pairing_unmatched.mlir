// RUN: ondrix-opt %s --pair-ondsp-fixed-reduction-outputs | FileCheck %s --implicit-check-not="lanes = 2"

// Every refusal below leaves the single-lane webs byte-identical: an
// unrecognized shape keeps the ordered schedule rather than a half-paired one.

// The differing streams are function arguments, so no pair table exists to
// interleave and the pairing has nothing to build from.

// CHECK-LABEL: func.func @refuse_non_constant_differing_stream
// CHECK-COUNT-2: ondsp.reduce_mac

func.func @refuse_non_constant_differing_stream(
    %arg0: memref<4xi16>, %arg1: memref<4xi16>, %arg2: memref<4xi16>) -> memref<4xi16> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %alloc = memref.alloc() {alignment = 64 : i64} : memref<4xi16>
  %0 = ondsp.acc_zero : <storage = i40, frac = 30, signed, update_overflow = saturate>
  %1 = ondsp.reduce_mac %0, %arg0, %arg1 {numeric = #ondsp.fixed<signed, storage = i16, frac = 15>, product = #ondsp.product<full>} : (!ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>, memref<4xi16>, memref<4xi16>) -> !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>
  %2 = ondsp.acc_export %1 {dst = #ondsp.fixed<signed, storage = i16, frac = 15>, overflow = #ondsp.overflow<saturate>, rounding = #ondsp.rounding<nearest_ties_positive>} : (!ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>) -> i16
  memref.store %2, %alloc[%c0] : memref<4xi16>
  %3 = ondsp.acc_zero : <storage = i40, frac = 30, signed, update_overflow = saturate>
  %4 = ondsp.reduce_mac %3, %arg0, %arg2 {numeric = #ondsp.fixed<signed, storage = i16, frac = 15>, product = #ondsp.product<full>} : (!ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>, memref<4xi16>, memref<4xi16>) -> !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>
  %5 = ondsp.acc_export %4 {dst = #ondsp.fixed<signed, storage = i16, frac = 15>, overflow = #ondsp.overflow<saturate>, rounding = #ondsp.rounding<nearest_ties_positive>} : (!ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>) -> i16
  memref.store %5, %alloc[%c1] : memref<4xi16>
  return %alloc : memref<4xi16>
}

memref.global "private" constant @policy_row_a : memref<4xi16> = dense<[1, 2, 3, 4]>
memref.global "private" constant @policy_row_b : memref<4xi16> = dense<[5, 6, 7, 8]>

// One dual-lane export carries one destination policy, so two webs that
// requantize differently cannot share it.

// CHECK-LABEL: func.func @refuse_differing_export_policy
// CHECK-COUNT-2: ondsp.reduce_mac

func.func @refuse_differing_export_policy(%arg0: memref<4xi16>) -> memref<4xi16> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %alloc = memref.alloc() {alignment = 64 : i64} : memref<4xi16>
  %0 = memref.get_global @policy_row_a : memref<4xi16>
  %1 = ondsp.acc_zero : <storage = i40, frac = 30, signed, update_overflow = saturate>
  %2 = ondsp.reduce_mac %1, %arg0, %0 {numeric = #ondsp.fixed<signed, storage = i16, frac = 15>, product = #ondsp.product<full>} : (!ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>, memref<4xi16>, memref<4xi16>) -> !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>
  %3 = ondsp.acc_export %2 {dst = #ondsp.fixed<signed, storage = i16, frac = 15>, overflow = #ondsp.overflow<saturate>, rounding = #ondsp.rounding<nearest_ties_positive>} : (!ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>) -> i16
  memref.store %3, %alloc[%c0] : memref<4xi16>
  %4 = memref.get_global @policy_row_b : memref<4xi16>
  %5 = ondsp.acc_zero : <storage = i40, frac = 30, signed, update_overflow = saturate>
  %6 = ondsp.reduce_mac %5, %arg0, %4 {numeric = #ondsp.fixed<signed, storage = i16, frac = 15>, product = #ondsp.product<full>} : (!ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>, memref<4xi16>, memref<4xi16>) -> !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>
  %7 = ondsp.acc_export %6 {dst = #ondsp.fixed<signed, storage = i16, frac = 15>, overflow = #ondsp.overflow<saturate>, rounding = #ondsp.rounding<nearest_even>} : (!ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>) -> i16
  memref.store %7, %alloc[%c1] : memref<4xi16>
  return %alloc : memref<4xi16>
}

memref.global "private" constant @alias_row_a : memref<4xi16> = dense<[1, 2, 3, 4]>
memref.global "private" constant @alias_row_b : memref<4xi16> = dense<[5, 6, 7, 8]>

// This witness discriminates the store-motion refusal alone: it is the pairable
// shape with the first web's store aimed at the very buffer both webs read.

// CHECK-LABEL: func.func @refuse_store_aliasing_shared_stream
// CHECK-COUNT-2: ondsp.reduce_mac

func.func @refuse_store_aliasing_shared_stream(%arg0: memref<4xi16>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %0 = memref.get_global @alias_row_a : memref<4xi16>
  %1 = ondsp.acc_zero : <storage = i40, frac = 30, signed, update_overflow = saturate>
  %2 = ondsp.reduce_mac %1, %arg0, %0 {numeric = #ondsp.fixed<signed, storage = i16, frac = 15>, product = #ondsp.product<full>} : (!ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>, memref<4xi16>, memref<4xi16>) -> !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>
  %3 = ondsp.acc_export %2 {dst = #ondsp.fixed<signed, storage = i16, frac = 15>, overflow = #ondsp.overflow<saturate>, rounding = #ondsp.rounding<nearest_ties_positive>} : (!ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>) -> i16
  memref.store %3, %arg0[%c0] : memref<4xi16>
  %4 = memref.get_global @alias_row_b : memref<4xi16>
  %5 = ondsp.acc_zero : <storage = i40, frac = 30, signed, update_overflow = saturate>
  %6 = ondsp.reduce_mac %5, %arg0, %4 {numeric = #ondsp.fixed<signed, storage = i16, frac = 15>, product = #ondsp.product<full>} : (!ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>, memref<4xi16>, memref<4xi16>) -> !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>
  %7 = ondsp.acc_export %6 {dst = #ondsp.fixed<signed, storage = i16, frac = 15>, overflow = #ondsp.overflow<saturate>, rounding = #ondsp.rounding<nearest_ties_positive>} : (!ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>) -> i16
  memref.store %7, %arg0[%c1] : memref<4xi16>
  return
}

memref.global "private" constant @spanned_row_a : memref<4xi16> = dense<[1, 2, 3, 4]>
memref.global "private" constant @spanned_row_b : memref<4xi16> = dense<[5, 6, 7, 8]>

// The pairing moves the second web's loads before the first web's store, so an
// unrelated write in that span is a reordering this cannot account for.

// CHECK-LABEL: func.func @refuse_effect_between_webs
// CHECK-COUNT-2: ondsp.reduce_mac

func.func @refuse_effect_between_webs(
    %arg0: memref<4xi16>, %arg1: memref<4xi16>) -> memref<4xi16> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %cst = arith.constant 3 : i16
  %alloc = memref.alloc() {alignment = 64 : i64} : memref<4xi16>
  %0 = memref.get_global @spanned_row_a : memref<4xi16>
  %1 = ondsp.acc_zero : <storage = i40, frac = 30, signed, update_overflow = saturate>
  %2 = ondsp.reduce_mac %1, %arg0, %0 {numeric = #ondsp.fixed<signed, storage = i16, frac = 15>, product = #ondsp.product<full>} : (!ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>, memref<4xi16>, memref<4xi16>) -> !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>
  %3 = ondsp.acc_export %2 {dst = #ondsp.fixed<signed, storage = i16, frac = 15>, overflow = #ondsp.overflow<saturate>, rounding = #ondsp.rounding<nearest_ties_positive>} : (!ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>) -> i16
  memref.store %3, %alloc[%c0] : memref<4xi16>
  memref.store %cst, %arg1[%c0] : memref<4xi16>
  %4 = memref.get_global @spanned_row_b : memref<4xi16>
  %5 = ondsp.acc_zero : <storage = i40, frac = 30, signed, update_overflow = saturate>
  %6 = ondsp.reduce_mac %5, %arg0, %4 {numeric = #ondsp.fixed<signed, storage = i16, frac = 15>, product = #ondsp.product<full>} : (!ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>, memref<4xi16>, memref<4xi16>) -> !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>
  %7 = ondsp.acc_export %6 {dst = #ondsp.fixed<signed, storage = i16, frac = 15>, overflow = #ondsp.overflow<saturate>, rounding = #ondsp.rounding<nearest_ties_positive>} : (!ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>) -> i16
  memref.store %7, %alloc[%c1] : memref<4xi16>
  return %alloc : memref<4xi16>
}

// An odd output count leaves one column without a partner, and splitting one
// output across the lanes is not an exact pairing.

// CHECK-LABEL: func.func @refuse_odd_column_count
// CHECK: memref.alloc() {alignment = 64 : i64} : memref<3x4xi16>
// CHECK: ondsp.reduce_mac

func.func @refuse_odd_column_count(%arg0: memref<4x4xi16>, %arg1: memref<4x3xi16>) -> memref<4x3xi16> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c3 = arith.constant 3 : index
  %c4 = arith.constant 4 : index
  %alloc = memref.alloc() {alignment = 64 : i64} : memref<4x3xi16>
  %alloc_0 = memref.alloc() {alignment = 64 : i64} : memref<3x4xi16>
  scf.for %arg2 = %c0 to %c3 step %c1 {
    scf.for %arg3 = %c0 to %c4 step %c1 {
      %0 = memref.load %arg1[%arg3, %arg2] : memref<4x3xi16>
      memref.store %0, %alloc_0[%arg2, %arg3] : memref<3x4xi16>
    }
  }
  scf.for %arg2 = %c0 to %c4 step %c1 {
    %subview = memref.subview %arg0[%arg2, 0] [1, 4] [1, 1] : memref<4x4xi16> to memref<4xi16, strided<[1], offset: ?>>
    scf.for %arg3 = %c0 to %c3 step %c1 {
      %subview_1 = memref.subview %alloc_0[%arg3, 0] [1, 4] [1, 1] : memref<3x4xi16> to memref<4xi16, strided<[1], offset: ?>>
      %0 = ondsp.acc_zero : <storage = i40, frac = 30, signed, update_overflow = saturate>
      %1 = ondsp.reduce_mac %0, %subview, %subview_1 {numeric = #ondsp.fixed<signed, storage = i16, frac = 15>, product = #ondsp.product<full>} : (!ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>, memref<4xi16, strided<[1], offset: ?>>, memref<4xi16, strided<[1], offset: ?>>) -> !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>
      %2 = ondsp.acc_export %1 {dst = #ondsp.fixed<signed, storage = i16, frac = 15>, overflow = #ondsp.overflow<saturate>, rounding = #ondsp.rounding<nearest_ties_positive>} : (!ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>) -> i16
      memref.store %2, %alloc[%arg2, %arg3] : memref<4x3xi16>
    }
  }
  memref.dealloc %alloc_0 : memref<3x4xi16>
  return %alloc : memref<4x3xi16>
}

// The pair buffer is a stack allocation, so its size is a legality condition:
// 8 pairs of 66 elements is 1056 bytes, one column past what the budget holds.

// CHECK-LABEL: func.func @refuse_oversize_pair_buffer
// CHECK: memref.alloc() {alignment = 64 : i64} : memref<16x33xi16>
// CHECK-NOT: memref.alloca
// CHECK: ondsp.reduce_mac

func.func @refuse_oversize_pair_buffer(%arg0: memref<2x33xi16>, %arg1: memref<33x16xi16>) -> memref<2x16xi16> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c16 = arith.constant 16 : index
  %c33 = arith.constant 33 : index
  %alloc = memref.alloc() {alignment = 64 : i64} : memref<2x16xi16>
  %alloc_0 = memref.alloc() {alignment = 64 : i64} : memref<16x33xi16>
  scf.for %arg2 = %c0 to %c16 step %c1 {
    scf.for %arg3 = %c0 to %c33 step %c1 {
      %0 = memref.load %arg1[%arg3, %arg2] : memref<33x16xi16>
      memref.store %0, %alloc_0[%arg2, %arg3] : memref<16x33xi16>
    }
  }
  scf.for %arg2 = %c0 to %c2 step %c1 {
    %subview = memref.subview %arg0[%arg2, 0] [1, 33] [1, 1] : memref<2x33xi16> to memref<33xi16, strided<[1], offset: ?>>
    scf.for %arg3 = %c0 to %c16 step %c1 {
      %subview_1 = memref.subview %alloc_0[%arg3, 0] [1, 33] [1, 1] : memref<16x33xi16> to memref<33xi16, strided<[1], offset: ?>>
      %0 = ondsp.acc_zero : <storage = i40, frac = 30, signed, update_overflow = saturate>
      %1 = ondsp.reduce_mac %0, %subview, %subview_1 {numeric = #ondsp.fixed<signed, storage = i16, frac = 15>, product = #ondsp.product<full>} : (!ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>, memref<33xi16, strided<[1], offset: ?>>, memref<33xi16, strided<[1], offset: ?>>) -> !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>
      %2 = ondsp.acc_export %1 {dst = #ondsp.fixed<signed, storage = i16, frac = 15>, overflow = #ondsp.overflow<saturate>, rounding = #ondsp.rounding<nearest_ties_positive>} : (!ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>) -> i16
      memref.store %2, %alloc[%arg2, %arg3] : memref<2x16xi16>
    }
  }
  memref.dealloc %alloc_0 : memref<16x33xi16>
  return %alloc : memref<2x16xi16>
}

// A user outside the recognized set leaves the pack buffer's contents partly
// unaccounted for, so the whole allocation keeps its transposed schedule.

// CHECK-LABEL: func.func @refuse_foreign_pack_buffer_user
// CHECK: memref.alloc() {alignment = 64 : i64} : memref<4x4xi16>
// CHECK: memref.alloc() {alignment = 64 : i64} : memref<4x4xi16>
// CHECK: ondsp.reduce_mac

func.func @refuse_foreign_pack_buffer_user(%arg0: memref<4x4xi16>, %arg1: memref<4x4xi16>) -> memref<4x4xi16> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %alloc = memref.alloc() {alignment = 64 : i64} : memref<4x4xi16>
  %alloc_0 = memref.alloc() {alignment = 64 : i64} : memref<4x4xi16>
  scf.for %arg2 = %c0 to %c4 step %c1 {
    scf.for %arg3 = %c0 to %c4 step %c1 {
      %0 = memref.load %arg1[%arg3, %arg2] : memref<4x4xi16>
      memref.store %0, %alloc_0[%arg2, %arg3] : memref<4x4xi16>
    }
  }
  %probe = memref.load %alloc_0[%c0, %c0] : memref<4x4xi16>
  memref.store %probe, %alloc[%c0, %c0] : memref<4x4xi16>
  scf.for %arg2 = %c0 to %c4 step %c1 {
    %subview = memref.subview %arg0[%arg2, 0] [1, 4] [1, 1] : memref<4x4xi16> to memref<4xi16, strided<[1], offset: ?>>
    scf.for %arg3 = %c0 to %c4 step %c1 {
      %subview_1 = memref.subview %alloc_0[%arg3, 0] [1, 4] [1, 1] : memref<4x4xi16> to memref<4xi16, strided<[1], offset: ?>>
      %0 = ondsp.acc_zero : <storage = i40, frac = 30, signed, update_overflow = saturate>
      %1 = ondsp.reduce_mac %0, %subview, %subview_1 {numeric = #ondsp.fixed<signed, storage = i16, frac = 15>, product = #ondsp.product<full>} : (!ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>, memref<4xi16, strided<[1], offset: ?>>, memref<4xi16, strided<[1], offset: ?>>) -> !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>
      %2 = ondsp.acc_export %1 {dst = #ondsp.fixed<signed, storage = i16, frac = 15>, overflow = #ondsp.overflow<saturate>, rounding = #ondsp.rounding<nearest_ties_positive>} : (!ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>) -> i16
      memref.store %2, %alloc[%arg2, %arg3] : memref<4x4xi16>
    }
  }
  memref.dealloc %alloc_0 : memref<4x4xi16>
  return %alloc : memref<4x4xi16>
}
