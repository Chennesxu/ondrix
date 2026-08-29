// RUN: ondrix-opt %s --pair-ondsp-fixed-reduction-outputs | FileCheck %s

// Pairing is order-preserving: each lane folds its own output in the declared
// index order, so no proof is needed (the pass description carries the argument).

memref.global "private" constant @__ondrix_dct4_row0 : memref<4xi16> = dense<[32767, 32767, 32767, 32767]>
memref.global "private" constant @__ondrix_dct4_row1 : memref<4xi16> = dense<[30274, 12540, -12540, -30274]>

// The two tables interleave element by element, and `alignment = 4` is what
// makes each lane pair one aligned 32-bit load.
// CHECK: memref.global "private" constant @__ondrix_dct4_row0_row1_pair : memref<8xi16> = dense<[32767, 30274, 32767, 12540, 32767, -12540, 32767, -30274]> {alignment = 4 : i64}
// CHECK-NOT: @__ondrix_dct4_row0 :
// CHECK-NOT: @__ondrix_dct4_row1 :

// CHECK-LABEL: func.func @pair_constant_table_outputs
// CHECK: %[[TABLE:.*]] = memref.get_global @__ondrix_dct4_row0_row1_pair : memref<8xi16>
// CHECK: ondsp.acc_zero : <storage = i40, frac = 30, signed, update_overflow = saturate, lanes = 2>
// The shared stream becomes the scalar coefficient; the two tables become lanes.
// CHECK: memref.load %arg0[%{{.*}}] : memref<4xi16>
// CHECK: memref.load %[[TABLE]]
// CHECK: memref.load %[[TABLE]]
// CHECK: vector.insert {{.*}} [0] : i16 into vector<2xi16>
// CHECK: vector.insert {{.*}} [1] : i16 into vector<2xi16>
// CHECK: ondsp.mac
// CHECK-SAME: lanes = 2>, vector<2xi16>, i16)
// CHECK-COUNT-3: ondsp.mac
// CHECK-NOT: ondsp.mac
// CHECK: ondsp.acc_export
// CHECK-SAME: lanes = 2>) -> vector<2xi16>
// CHECK: %[[LANE0:.*]] = vector.extract %{{.*}}[0] : vector<2xi16>
// CHECK: memref.store %[[LANE0]], %[[OUT:.*]][%[[C0:.*]]]
// CHECK: %[[LANE1:.*]] = vector.extract %{{.*}}[1] : vector<2xi16>
// CHECK: memref.store %[[LANE1]], %[[OUT]][%[[C1:.*]]]
// CHECK-NOT: ondsp.reduce_mac

func.func @pair_constant_table_outputs(%arg0: memref<4xi16>) -> memref<4xi16> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %alloc = memref.alloc() {alignment = 64 : i64} : memref<4xi16>
  %0 = memref.get_global @__ondrix_dct4_row0 : memref<4xi16>
  %1 = ondsp.acc_zero : <storage = i40, frac = 30, signed, update_overflow = saturate>
  %2 = ondsp.reduce_mac %1, %arg0, %0 {numeric = #ondsp.fixed<signed, storage = i16, frac = 15>, product = #ondsp.product<full>} : (!ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>, memref<4xi16>, memref<4xi16>) -> !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>
  %3 = ondsp.acc_export %2 {dst = #ondsp.fixed<signed, storage = i16, frac = 11>, overflow = #ondsp.overflow<saturate>, rounding = #ondsp.rounding<nearest_ties_positive>} : (!ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>) -> i16
  memref.store %3, %alloc[%c0] : memref<4xi16>
  %4 = memref.get_global @__ondrix_dct4_row1 : memref<4xi16>
  %5 = ondsp.acc_zero : <storage = i40, frac = 30, signed, update_overflow = saturate>
  %6 = ondsp.reduce_mac %5, %arg0, %4 {numeric = #ondsp.fixed<signed, storage = i16, frac = 15>, product = #ondsp.product<full>} : (!ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>, memref<4xi16>, memref<4xi16>) -> !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>
  %7 = ondsp.acc_export %6 {dst = #ondsp.fixed<signed, storage = i16, frac = 11>, overflow = #ondsp.overflow<saturate>, rounding = #ondsp.rounding<nearest_ties_positive>} : (!ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>) -> i16
  memref.store %7, %alloc[%c1] : memref<4xi16>
  return %alloc : memref<4xi16>
}

// The transposed pack buffer is replaced by one that writes adjacent output
// columns interleaved, read straight from the source and never through the old one.

// CHECK-LABEL: func.func @pair_transposed_pack_columns
// A rank-1 stack allocation carrying the alignment is the load-bearing pair:
// the declared alignment reaches the backend only on a stack base.
// CHECK: %[[PAIRED:.*]] = memref.alloca() {alignment = 4 : i64} : memref<16xi16>
// CHECK: scf.for %[[P:.*]] = %{{.*}} to %{{.*}} step %{{.*}} {
// CHECK: scf.for %[[K:.*]] = %{{.*}} to %{{.*}} step %{{.*}} {
// CHECK: %[[STRIDE:.*]] = arith.constant 8 : index
// CHECK: %[[BASE:.*]] = arith.muli %[[P]], %[[STRIDE]] : index
// CHECK: %[[LOWSLOT:.*]] = arith.addi %[[BASE]], %{{.*}} : index
// CHECK: %[[HIGHSLOT:.*]] = arith.addi %[[LOWSLOT]], %{{.*}} : index
// CHECK: memref.load %arg1[%[[K]], %{{.*}}] : memref<4x4xi16>
// CHECK: memref.store %{{.*}}, %[[PAIRED]][%[[LOWSLOT]]] : memref<16xi16>
// CHECK: memref.load %arg1[%[[K]], %{{.*}}] : memref<4x4xi16>
// CHECK: memref.store %{{.*}}, %[[PAIRED]][%[[HIGHSLOT]]] : memref<16xi16>
// CHECK-NOT: memref.alloc() {alignment = 64 : i64} : memref<4x4xi16>
// The column loop is gone: two straight-line pair webs, each storing a constant
// column pair of the row the outer loop selects.
// CHECK: scf.for %[[ROW:.*]] = %{{.*}} to %{{.*}} step %{{.*}} {
// CHECK: %[[SHARED:.*]] = memref.subview %arg0[%[[ROW]], 0]
// CHECK: ondsp.acc_zero : <storage = i40, frac = 30, signed, update_overflow = saturate, lanes = 2>
// CHECK: memref.load %[[SHARED]]
// The unrolled pair walk leaves each lane load a literal slot of the flat buffer.
// CHECK: memref.load %[[PAIRED]][%{{.*}}] : memref<16xi16>
// CHECK: memref.load %[[PAIRED]][%{{.*}}] : memref<16xi16>
// CHECK: ondsp.acc_export
// CHECK-SAME: lanes = 2>) -> vector<2xi16>
// CHECK: memref.store %{{.*}}, %{{.*}}[%[[ROW]], %[[COL0:.*]]] : memref<4x4xi16>
// CHECK: memref.store %{{.*}}, %{{.*}}[%[[ROW]], %[[COL1:.*]]] : memref<4x4xi16>
// CHECK: ondsp.acc_zero : <storage = i40, frac = 30, signed, update_overflow = saturate, lanes = 2>
// CHECK: memref.store %{{.*}}, %{{.*}}[%[[ROW]], %[[COL2:.*]]] : memref<4x4xi16>
// CHECK: memref.store %{{.*}}, %{{.*}}[%[[ROW]], %[[COL3:.*]]] : memref<4x4xi16>
// CHECK-NOT: ondsp.reduce_mac
// The stack buffer needs no deallocation, and the transposed one is gone.
// CHECK-NOT: memref.dealloc

func.func @pair_transposed_pack_columns(%arg0: memref<4x4xi16>, %arg1: memref<4x4xi16>) -> memref<4x4xi16> {
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

memref.global "private" constant @tap_a : memref<4xi16> = dense<[1, 2, 3, 4]>
memref.global "private" constant @tap_b : memref<4xi16> = dense<[5, 6, 7, 8]>
memref.global "private" constant @tap_c : memref<4xi16> = dense<[9, 10, 11, 12]>

// An odd web count leaves the trailing web alone: pairing is a choice on
// adjacent outputs, and a leftover keeps the single-lane schedule.

// CHECK: memref.global "private" constant @tap_a_b_pair
// CHECK: memref.global "private" constant @tap_c
// CHECK-LABEL: func.func @leave_odd_web_unpaired
// CHECK: ondsp.acc_zero : <storage = i40, frac = 30, signed, update_overflow = saturate, lanes = 2>
// CHECK: ondsp.acc_export
// CHECK-SAME: lanes = 2>) -> vector<2xi16>
// CHECK: %[[TAIL:.*]] = memref.get_global @tap_c : memref<4xi16>
// CHECK: ondsp.acc_zero : <storage = i40, frac = 30, signed, update_overflow = saturate>
// CHECK: ondsp.reduce_mac %{{.*}}, %arg0, %[[TAIL]]
// CHECK: ondsp.acc_export
// CHECK-SAME: update_overflow = saturate>) -> i16

func.func @leave_odd_web_unpaired(%arg0: memref<4xi16>) -> memref<4xi16> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %alloc = memref.alloc() {alignment = 64 : i64} : memref<4xi16>
  %0 = memref.get_global @tap_a : memref<4xi16>
  %1 = ondsp.acc_zero : <storage = i40, frac = 30, signed, update_overflow = saturate>
  %2 = ondsp.reduce_mac %1, %arg0, %0 {numeric = #ondsp.fixed<signed, storage = i16, frac = 15>, product = #ondsp.product<full>} : (!ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>, memref<4xi16>, memref<4xi16>) -> !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>
  %3 = ondsp.acc_export %2 {dst = #ondsp.fixed<signed, storage = i16, frac = 15>, overflow = #ondsp.overflow<saturate>, rounding = #ondsp.rounding<nearest_ties_positive>} : (!ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>) -> i16
  memref.store %3, %alloc[%c0] : memref<4xi16>
  %4 = memref.get_global @tap_b : memref<4xi16>
  %5 = ondsp.acc_zero : <storage = i40, frac = 30, signed, update_overflow = saturate>
  %6 = ondsp.reduce_mac %5, %arg0, %4 {numeric = #ondsp.fixed<signed, storage = i16, frac = 15>, product = #ondsp.product<full>} : (!ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>, memref<4xi16>, memref<4xi16>) -> !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>
  %7 = ondsp.acc_export %6 {dst = #ondsp.fixed<signed, storage = i16, frac = 15>, overflow = #ondsp.overflow<saturate>, rounding = #ondsp.rounding<nearest_ties_positive>} : (!ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>) -> i16
  memref.store %7, %alloc[%c1] : memref<4xi16>
  %8 = memref.get_global @tap_c : memref<4xi16>
  %9 = ondsp.acc_zero : <storage = i40, frac = 30, signed, update_overflow = saturate>
  %10 = ondsp.reduce_mac %9, %arg0, %8 {numeric = #ondsp.fixed<signed, storage = i16, frac = 15>, product = #ondsp.product<full>} : (!ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>, memref<4xi16>, memref<4xi16>) -> !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>
  %11 = ondsp.acc_export %10 {dst = #ondsp.fixed<signed, storage = i16, frac = 15>, overflow = #ondsp.overflow<saturate>, rounding = #ondsp.rounding<nearest_ties_positive>} : (!ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>) -> i16
  memref.store %11, %alloc[%c2] : memref<4xi16>
  return %alloc : memref<4xi16>
}
