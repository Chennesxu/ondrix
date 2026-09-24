// RUN: ondrix-opt %s --lower-ondsp-assumptions | FileCheck %s --check-prefix=TRUST
// RUN: ondrix-opt %s --lower-ondsp-assumptions=checked=true | FileCheck %s --check-prefix=CHECK

// Trusted, the declaration costs nothing; checked, it sums the absolute raw
// values in i64 and aborts unless the sum stays below the bound.
// TRUST-LABEL: func.func @q15_kernel
// TRUST-NOT: ondsp.assume_l1_bound
// TRUST-NOT: cf.assert
// TRUST: memref.load %arg0
// CHECK-LABEL: func.func @q15_kernel
// CHECK: scf.for {{.*}} iter_args(%{{.*}} = %{{.*}}) -> (i64)
// CHECK: arith.maxsi
// CHECK: %[[BOUND:.*]] = arith.constant 65536 : i64
// CHECK: %[[HOLDS:.*]] = arith.cmpi slt, %{{.*}}, %[[BOUND]] : i64
// CHECK: cf.assert %[[HOLDS]], "ondrix_q15_kernel: coefficients exceed the declared gain_bound"
// CHECK-NOT: ondsp.assume_l1_bound
// CHECK: memref.load %arg0
func.func @q15_kernel(%taps: memref<?xi16>) -> i16 {
  %c0 = arith.constant 0 : index
  %bounded = ondsp.assume_l1_bound %taps {bound = 65536 : i64} : memref<?xi16>
  %first = memref.load %bounded[%c0] : memref<?xi16>
  return %first : i16
}

// Trusted, the symbol bound costs nothing; checked, every magnitude is tested.
// TRUST-LABEL: func.func @q15_decoder
// TRUST-NOT: cf.assert
// CHECK-LABEL: func.func @q15_decoder
// CHECK: %[[BOUND:.*]] = arith.constant 127 : i64
// CHECK: scf.for
// CHECK: arith.maxsi
// CHECK: %[[HOLDS:.*]] = arith.cmpi sle, %{{.*}}, %[[BOUND]] : i64
// CHECK: cf.assert %[[HOLDS]], "ondrix_q15_decoder: symbols exceed the declared symbol_bound"
// CHECK-NOT: ondsp.assume_magnitude_bound
func.func @q15_decoder(%symbols: memref<?xi16>) -> i16 {
  %c0 = arith.constant 0 : index
  %bounded = ondsp.assume_magnitude_bound %symbols {bound = 127 : i64} : memref<?xi16>
  %first = memref.load %bounded[%c0] : memref<?xi16>
  return %first : i16
}
