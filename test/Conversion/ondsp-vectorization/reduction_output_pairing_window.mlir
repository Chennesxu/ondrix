// RUN: ondrix-opt %s --pair-ondsp-fixed-reduction-outputs | FileCheck %s

// The window-loop matcher: outputs n and n+1 of a sliding-window loop become
// one dual-lane web whose scalar coefficient is the K+1-sample span both
// windows cover and whose lanes read the tap pair buffer.

// CHECK-LABEL: func.func @pair_window_loop_odd
// The pair buffer holds 2(K+1) taps: slot 2j is tap j, slot 2j+1 is tap j-1,
// and the two out-of-range taps are the exact zero.
// CHECK: %[[PAIRS:.*]] = memref.alloca() {alignment = 4 : i64} : memref<10xi16>
// CHECK: %[[ZERO:.*]] = arith.constant 0 : i16
// CHECK: memref.store %[[ZERO]], %[[PAIRS]][%[[SLOT1:.*]]] : memref<10xi16>
// CHECK: %[[SLOT8:.*]] = arith.constant 8 : index
// CHECK: memref.store %[[ZERO]], %[[PAIRS]][%[[SLOT8]]] : memref<10xi16>
// CHECK: %[[TAPS:.*]] = arith.constant 4 : index
// CHECK: scf.for %[[TAP:.*]] = %{{.*}} to %[[TAPS]]
// CHECK: %[[C:.*]] = memref.load %{{.*}}[%[[TAP]]]
// CHECK: memref.store %[[C]], %[[PAIRS]]
// CHECK: memref.store %[[C]], %[[PAIRS]]
// Four pairs, each over a five-sample window with five dual steps.
// CHECK: memref.subview %arg0[0] [5] [1] : memref<12xi16> to memref<5xi16
// CHECK: ondsp.acc_zero : <storage = i40, frac = 30, signed, update_overflow = saturate, lanes = 2>
// CHECK-COUNT-5: ondsp.mac
// CHECK: ondsp.acc_export
// CHECK-SAME: lanes = 2>) -> vector<2xi16>
// CHECK: vector.extract %{{.*}}[0] : vector<2xi16>
// CHECK: memref.store %{{.*}}, %[[OUT:.*]][%{{.*}}] : memref<9xi16>
// CHECK: vector.extract %{{.*}}[1] : vector<2xi16>
// CHECK: memref.store %{{.*}}, %[[OUT]][%{{.*}}] : memref<9xi16>
// CHECK: memref.subview %arg0[2] [5] [1]
// CHECK: memref.subview %arg0[4] [5] [1]
// CHECK: memref.subview %arg0[6] [5] [1]
// The ninth output has no partner and stays on the single-lane loop.
// CHECK: scf.for %[[N:.*]] = %c8{{[_0-9]*}} to %c9 step
// CHECK: memref.subview %arg0[%[[N]]] [4] [1]
// CHECK: ondsp.reduce_mac
func.func @pair_window_loop_odd(%arg0: memref<12xi16>, %arg1: memref<4xi16>) -> memref<9xi16> {
  %c9 = arith.constant 9 : index
  %c1 = arith.constant 1 : index
  %c0 = arith.constant 0 : index
  %alloc = memref.alloc() {alignment = 64 : i64} : memref<9xi16>
  %cast = memref.cast %arg1 : memref<4xi16> to memref<?xi16, strided<[1]>>
  scf.for %n = %c0 to %c9 step %c1 {
    %subview = memref.subview %arg0[%n] [4] [1] : memref<12xi16> to memref<4xi16, strided<[1], offset: ?>>
    %cast_0 = memref.cast %subview : memref<4xi16, strided<[1], offset: ?>> to memref<?xi16, strided<[1], offset: ?>>
    %0 = ondsp.acc_zero : <storage = i40, frac = 30, signed, update_overflow = saturate>
    %1 = ondsp.reduce_mac %0, %cast_0, %cast {numeric = #ondsp.fixed<signed, storage = i16, frac = 15>, product = #ondsp.product<full>} : (!ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>, memref<?xi16, strided<[1], offset: ?>>, memref<?xi16, strided<[1]>>) -> !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>
    %2 = ondsp.acc_export %1 {dst = #ondsp.fixed<signed, storage = i16, frac = 15>, overflow = #ondsp.overflow<saturate>, rounding = #ondsp.rounding<nearest_ties_positive>} : (!ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>) -> i16
    memref.store %2, %alloc[%n] : memref<9xi16>
  }
  return %alloc : memref<9xi16>
}

// An even output count pairs completely and the original loop goes; above the
// straight-line bound the pair walk stays a loop.
// CHECK-LABEL: func.func @pair_window_loop_even_long
// CHECK: memref.alloca() {alignment = 4 : i64} : memref<66xi16>
// CHECK: %[[PAIRCOUNT:.*]] = arith.constant 17 : index
// CHECK: scf.for %[[P:.*]] = %{{.*}} to %[[PAIRCOUNT]]
// CHECK: %[[FIRST:.*]] = arith.muli %[[P]], %{{.*}} : index
// CHECK: memref.subview %arg0[%[[FIRST]]] [33] [1]
// CHECK-COUNT-33: ondsp.mac
// CHECK-NOT: ondsp.reduce_mac
func.func @pair_window_loop_even_long(%arg0: memref<65xi16>, %arg1: memref<32xi16>) -> memref<34xi16> {
  %c34 = arith.constant 34 : index
  %c1 = arith.constant 1 : index
  %c0 = arith.constant 0 : index
  %alloc = memref.alloc() {alignment = 64 : i64} : memref<34xi16>
  scf.for %n = %c0 to %c34 step %c1 {
    %subview = memref.subview %arg0[%n] [32] [1] : memref<65xi16> to memref<32xi16, strided<[1], offset: ?>>
    %0 = ondsp.acc_zero : <storage = i40, frac = 30, signed, update_overflow = saturate>
    %1 = ondsp.reduce_mac %0, %subview, %arg1 {numeric = #ondsp.fixed<signed, storage = i16, frac = 15>, product = #ondsp.product<full>} : (!ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>, memref<32xi16, strided<[1], offset: ?>>, memref<32xi16>) -> !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>
    %2 = ondsp.acc_export %1 {dst = #ondsp.fixed<signed, storage = i16, frac = 15>, overflow = #ondsp.overflow<saturate>, rounding = #ondsp.rounding<nearest_ties_positive>} : (!ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>) -> i16
    memref.store %2, %alloc[%n] : memref<34xi16>
  }
  return %alloc : memref<34xi16>
}
