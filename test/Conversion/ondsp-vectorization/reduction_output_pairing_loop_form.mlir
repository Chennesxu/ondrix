// RUN: ondrix-opt %s --pair-ondsp-fixed-reduction-outputs | FileCheck %s

// Above the straight-line bound the paired chain stays a loop whose single
// iteration argument is the dual-lane accumulator.

memref.global "private" constant @long_row_a : memref<96xi16> = dense<7>
memref.global "private" constant @long_row_b : memref<96xi16> = dense<-3>

// A long dense table prints little-endian hex: 0x0700 is 7 and 0xFDFF is -3,
// so the leading bytes show the interleave the pair table carries.
// CHECK: memref.global "private" constant @long_row_a_b_pair : memref<192xi16> = dense<"0x0700FDFF0700FDFF{{.*}}"> {alignment = 4 : i64}

// CHECK-LABEL: func.func @loop_long_reduction_pair
// CHECK: %[[TABLE:.*]] = memref.get_global @long_row_a_b_pair : memref<192xi16>
// CHECK: %[[INIT:.*]] = ondsp.acc_zero : <storage = i40, frac = 30, signed, update_overflow = saturate, lanes = 2>
// CHECK: %[[SUM:.*]] = scf.for %[[K:.*]] = %{{.*}} to %{{.*}} step %{{.*}} iter_args(%[[ACC:.*]] = %[[INIT]]) -> (!ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate, lanes = 2>) {
// CHECK: memref.load %arg0[%[[K]]] : memref<96xi16>
// CHECK: %[[LOW:.*]] = arith.muli %[[K]], %{{.*}} : index
// CHECK: %[[HIGH:.*]] = arith.addi %[[LOW]], %{{.*}} : index
// CHECK: memref.load %[[TABLE]][%[[LOW]]] : memref<192xi16>
// CHECK: memref.load %[[TABLE]][%[[HIGH]]] : memref<192xi16>
// CHECK: %[[NEXT:.*]] = ondsp.mac %[[ACC]]
// CHECK-SAME: lanes = 2>, vector<2xi16>, i16)
// CHECK: scf.yield %[[NEXT]]
// CHECK: ondsp.acc_export %[[SUM]]
// CHECK-SAME: lanes = 2>) -> vector<2xi16>
// CHECK: vector.extract %{{.*}}[0] : vector<2xi16>
// CHECK: vector.extract %{{.*}}[1] : vector<2xi16>

func.func @loop_long_reduction_pair(%arg0: memref<96xi16>) -> memref<2xi16> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %alloc = memref.alloc() {alignment = 64 : i64} : memref<2xi16>
  %0 = memref.get_global @long_row_a : memref<96xi16>
  %1 = ondsp.acc_zero : <storage = i40, frac = 30, signed, update_overflow = saturate>
  %2 = ondsp.reduce_mac %1, %arg0, %0 {numeric = #ondsp.fixed<signed, storage = i16, frac = 15>, product = #ondsp.product<full>} : (!ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>, memref<96xi16>, memref<96xi16>) -> !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>
  %3 = ondsp.acc_export %2 {dst = #ondsp.fixed<signed, storage = i16, frac = 15>, overflow = #ondsp.overflow<saturate>, rounding = #ondsp.rounding<nearest_ties_positive>} : (!ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>) -> i16
  memref.store %3, %alloc[%c0] : memref<2xi16>
  %4 = memref.get_global @long_row_b : memref<96xi16>
  %5 = ondsp.acc_zero : <storage = i40, frac = 30, signed, update_overflow = saturate>
  %6 = ondsp.reduce_mac %5, %arg0, %4 {numeric = #ondsp.fixed<signed, storage = i16, frac = 15>, product = #ondsp.product<full>} : (!ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>, memref<96xi16>, memref<96xi16>) -> !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>
  %7 = ondsp.acc_export %6 {dst = #ondsp.fixed<signed, storage = i16, frac = 15>, overflow = #ondsp.overflow<saturate>, rounding = #ondsp.rounding<nearest_ties_positive>} : (!ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>) -> i16
  memref.store %7, %alloc[%c1] : memref<2xi16>
  return %alloc : memref<2xi16>
}
