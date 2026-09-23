// RUN: ondrix-opt %s --split-input-file --verify-diagnostics | FileCheck %s

// The declaration returns its source unchanged and prints back.
// CHECK-LABEL: func.func @declared
// CHECK: ondsp.assume_l1_bound %arg0 {bound = 65536 : i64} : memref<16xi16>
func.func @declared(%taps: memref<16xi16>) -> memref<16xi16> {
  %bounded = ondsp.assume_l1_bound %taps {bound = 65536 : i64} : memref<16xi16>
  return %bounded : memref<16xi16>
}

// -----

func.func @empty_bound(%taps: memref<16xi16>) -> memref<16xi16> {
  // expected-error @+1 {{bound must be positive}}
  %bounded = ondsp.assume_l1_bound %taps {bound = 0 : i64} : memref<16xi16>
  return %bounded : memref<16xi16>
}
