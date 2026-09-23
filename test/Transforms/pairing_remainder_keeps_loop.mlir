// RUN: ondrix-opt %s --scalarize-ondsp-fixed-reduce-mac --split-input-file | FileCheck %s --check-prefix=SCALARIZE
// RUN: ondrix-opt %s --unroll-ondsp-fixed-mac-loops --split-input-file | FileCheck %s --check-prefix=UNROLL

// Output pairing's remainder keeps its loop under both passes and every budget;
// the unmarked twin of each shape straight-lines, which is what the mark changes.

// SCALARIZE-LABEL: func.func @marked_reduction
// SCALARIZE: scf.for {{.*}} iter_args
func.func @marked_reduction(%x: memref<12xi16>, %taps: memref<4xi16>, %out: memref<9xi16>) {
  %c8 = arith.constant 8 : index
  %c9 = arith.constant 9 : index
  %c1 = arith.constant 1 : index
  scf.for %n = %c8 to %c9 step %c1 {
    %window = memref.subview %x[%n] [4] [1] : memref<12xi16> to memref<4xi16, strided<[1], offset: ?>>
    %zero = ondsp.acc_zero : !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>
    %sum = ondsp.reduce_mac %zero, %window, %taps {numeric = #ondsp.fixed<signed, storage = i16, frac = 15>, product = #ondsp.product<full>} : (!ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>, memref<4xi16, strided<[1], offset: ?>>, memref<4xi16>) -> !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>
    %value = ondsp.acc_export %sum {dst = #ondsp.fixed<signed, storage = i16, frac = 15>, overflow = #ondsp.overflow<saturate>, rounding = #ondsp.rounding<nearest_ties_positive>} : (!ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>) -> i16
    memref.store %value, %out[%n] : memref<9xi16>
  } {ondsp.pairing_remainder}
  return
}

// -----

// SCALARIZE-LABEL: func.func @unmarked_reduction
// SCALARIZE-NOT: iter_args
// SCALARIZE: return
func.func @unmarked_reduction(%x: memref<12xi16>, %taps: memref<4xi16>, %out: memref<9xi16>) {
  %c8 = arith.constant 8 : index
  %c9 = arith.constant 9 : index
  %c1 = arith.constant 1 : index
  scf.for %n = %c8 to %c9 step %c1 {
    %window = memref.subview %x[%n] [4] [1] : memref<12xi16> to memref<4xi16, strided<[1], offset: ?>>
    %zero = ondsp.acc_zero : !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>
    %sum = ondsp.reduce_mac %zero, %window, %taps {numeric = #ondsp.fixed<signed, storage = i16, frac = 15>, product = #ondsp.product<full>} : (!ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>, memref<4xi16, strided<[1], offset: ?>>, memref<4xi16>) -> !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>
    %value = ondsp.acc_export %sum {dst = #ondsp.fixed<signed, storage = i16, frac = 15>, overflow = #ondsp.overflow<saturate>, rounding = #ondsp.rounding<nearest_ties_positive>} : (!ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>) -> i16
    memref.store %value, %out[%n] : memref<9xi16>
  }
  return
}

// -----

// UNROLL-LABEL: func.func @marked_loop
// UNROLL: scf.for {{.*}} iter_args
func.func @marked_loop(%x: memref<12xi16>, %taps: memref<4xi16>, %out: memref<9xi16>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %c8 = arith.constant 8 : index
  %c9 = arith.constant 9 : index
  scf.for %n = %c8 to %c9 step %c1 {
    %zero = ondsp.acc_zero : !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>
    %sum = scf.for %k = %c0 to %c4 step %c1 iter_args(%acc = %zero) -> (!ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>) {
      %index = arith.addi %n, %k : index
      %a = memref.load %x[%index] : memref<12xi16>
      %b = memref.load %taps[%k] : memref<4xi16>
      %next = ondsp.mac %acc, %a, %b {numeric = #ondsp.fixed<signed, storage = i16, frac = 15>, product = #ondsp.product<full>} : (!ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>, i16, i16) -> !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>
      scf.yield %next : !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>
    }
    %value = ondsp.acc_export %sum {dst = #ondsp.fixed<signed, storage = i16, frac = 15>, overflow = #ondsp.overflow<saturate>, rounding = #ondsp.rounding<nearest_ties_positive>} : (!ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>) -> i16
    memref.store %value, %out[%n] : memref<9xi16>
  } {ondsp.pairing_remainder}
  return
}

// -----

// UNROLL-LABEL: func.func @unmarked_loop
// UNROLL-NOT: iter_args
// UNROLL: return
func.func @unmarked_loop(%x: memref<12xi16>, %taps: memref<4xi16>, %out: memref<9xi16>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %c8 = arith.constant 8 : index
  %c9 = arith.constant 9 : index
  scf.for %n = %c8 to %c9 step %c1 {
    %zero = ondsp.acc_zero : !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>
    %sum = scf.for %k = %c0 to %c4 step %c1 iter_args(%acc = %zero) -> (!ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>) {
      %index = arith.addi %n, %k : index
      %a = memref.load %x[%index] : memref<12xi16>
      %b = memref.load %taps[%k] : memref<4xi16>
      %next = ondsp.mac %acc, %a, %b {numeric = #ondsp.fixed<signed, storage = i16, frac = 15>, product = #ondsp.product<full>} : (!ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>, i16, i16) -> !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>
      scf.yield %next : !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>
    }
    %value = ondsp.acc_export %sum {dst = #ondsp.fixed<signed, storage = i16, frac = 15>, overflow = #ondsp.overflow<saturate>, rounding = #ondsp.rounding<nearest_ties_positive>} : (!ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>) -> i16
    memref.store %value, %out[%n] : memref<9xi16>
  }
  return
}
