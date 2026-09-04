// RUN: ondrix-opt %s --vectorize-ondsp-fixed-decimate-outputs=vector-width=4 | FileCheck %s

// Width 4 over eight columns: two blocks read B row segments directly, the
// ordered loop and the packed copy that fed it are gone.
// CHECK-LABEL: func.func @matmul_columns
// CHECK-NOT: memref.alloc() {alignment = 64 : i64} : memref<8x8xi16>
// CHECK: %[[OUT:.*]] = memref.alloc()
// CHECK-NOT: memref.alloc
// CHECK: scf.for %[[ROW:.*]] = %{{.*}} to %{{.*}} step %{{.*}} {
// CHECK: %[[LHS:.*]] = memref.subview %{{.*}}[%[[ROW]], 0] [1, 8] [1, 1]
// CHECK: scf.for %[[BLOCK:.*]] = %{{.*}} to %{{.*}} step %{{.*}} {
// CHECK: ondsp.acc_zero : <storage = i40, frac = 30, signed, update_overflow = wrap, lanes = 4>
// CHECK: vector.load %{{.*}}[%{{.*}}, %[[BLOCK]]] : memref<8x8xi16>, vector<4xi16>
// CHECK: memref.load %[[LHS]][%{{.*}}]
// CHECK-COUNT-8: ondsp.mac
// CHECK: ondsp.acc_export {{.*}} -> vector<4xi16>
// CHECK: vector.store %{{.*}}, %[[OUT]][%[[ROW]], %[[BLOCK]]] : memref<8x8xi16>, vector<4xi16>
// CHECK-NOT: ondsp.reduce_mac
// CHECK-NOT: memref.dealloc
func.func @matmul_columns(%a: memref<8x8xi16>, %b: memref<8x8xi16>) -> memref<8x8xi16> {
  %c8 = arith.constant 8 : index
  %c1 = arith.constant 1 : index
  %c0 = arith.constant 0 : index
  %alloc = memref.alloc() {alignment = 64 : i64} : memref<8x8xi16>
  %packed = memref.alloc() {alignment = 64 : i64} : memref<8x8xi16>
  scf.for %c = %c0 to %c8 step %c1 {
    scf.for %k = %c0 to %c8 step %c1 {
      %0 = memref.load %b[%k, %c] : memref<8x8xi16>
      memref.store %0, %packed[%c, %k] : memref<8x8xi16>
    }
  }
  scf.for %i = %c0 to %c8 step %c1 {
    %row = memref.subview %a[%i, 0] [1, 8] [1, 1] : memref<8x8xi16> to memref<8xi16, strided<[1], offset: ?>>
    scf.for %j = %c0 to %c8 step %c1 {
      %column = memref.subview %packed[%j, 0] [1, 8] [1, 1] : memref<8x8xi16> to memref<8xi16, strided<[1], offset: ?>>
      %0 = ondsp.acc_zero : <storage = i40, frac = 30, signed, update_overflow = wrap>
      %1 = ondsp.reduce_mac %0, %row, %column {numeric = #ondsp.fixed<signed, storage = i16, frac = 15>, product = #ondsp.product<full>} : (!ondsp.acc<storage = i40, frac = 30, signed, update_overflow = wrap>, memref<8xi16, strided<[1], offset: ?>>, memref<8xi16, strided<[1], offset: ?>>) -> !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = wrap>
      %2 = ondsp.acc_export %1 {dst = #ondsp.fixed<signed, storage = i16, frac = 15>, overflow = #ondsp.overflow<saturate>, rounding = #ondsp.rounding<nearest_even>} : (!ondsp.acc<storage = i40, frac = 30, signed, update_overflow = wrap>) -> i16
      memref.store %2, %alloc[%i, %j] : memref<8x8xi16>
    }
  }
  memref.dealloc %packed : memref<8x8xi16>
  return %alloc : memref<8x8xi16>
}

// Three columns under width 4: one four-lane block takes every column; the
// last row's segment is loaded narrow and zero-padded, and only three lanes store.
// CHECK-LABEL: func.func @matmul_narrow_columns
// CHECK: ondsp.acc_zero : <storage = i40, frac = 30, signed, update_overflow = wrap, lanes = 4>
// CHECK: vector.load %{{.*}} : memref<16x3xi16>, vector<4xi16>
// CHECK-COUNT-15: ondsp.mac
// CHECK: vector.load %{{.*}} : memref<16x3xi16>, vector<3xi16>
// CHECK: vector.shuffle {{.*}} [0, 1, 2, 3] : vector<3xi16>, vector<3xi16>
// CHECK: ondsp.mac
// CHECK: vector.extract_strided_slice {{.*}} {offsets = [0], sizes = [3], strides = [1]}
// CHECK: vector.store %{{.*}} : memref<4x3xi16>, vector<3xi16>
// CHECK-NOT: ondsp.reduce_mac
func.func @matmul_narrow_columns(%a: memref<4x16xi16>, %b: memref<16x3xi16>) -> memref<4x3xi16> {
  %c3 = arith.constant 3 : index
  %c16 = arith.constant 16 : index
  %c4 = arith.constant 4 : index
  %c1 = arith.constant 1 : index
  %c0 = arith.constant 0 : index
  %alloc = memref.alloc() {alignment = 64 : i64} : memref<4x3xi16>
  %packed = memref.alloc() {alignment = 64 : i64} : memref<3x16xi16>
  scf.for %c = %c0 to %c3 step %c1 {
    scf.for %k = %c0 to %c16 step %c1 {
      %0 = memref.load %b[%k, %c] : memref<16x3xi16>
      memref.store %0, %packed[%c, %k] : memref<3x16xi16>
    }
  }
  scf.for %i = %c0 to %c4 step %c1 {
    %row = memref.subview %a[%i, 0] [1, 16] [1, 1] : memref<4x16xi16> to memref<16xi16, strided<[1], offset: ?>>
    scf.for %j = %c0 to %c3 step %c1 {
      %column = memref.subview %packed[%j, 0] [1, 16] [1, 1] : memref<3x16xi16> to memref<16xi16, strided<[1], offset: ?>>
      %0 = ondsp.acc_zero : <storage = i40, frac = 30, signed, update_overflow = wrap>
      %1 = ondsp.reduce_mac %0, %row, %column {numeric = #ondsp.fixed<signed, storage = i16, frac = 15>, product = #ondsp.product<full>} : (!ondsp.acc<storage = i40, frac = 30, signed, update_overflow = wrap>, memref<16xi16, strided<[1], offset: ?>>, memref<16xi16, strided<[1], offset: ?>>) -> !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = wrap>
      %2 = ondsp.acc_export %1 {dst = #ondsp.fixed<signed, storage = i16, frac = 15>, overflow = #ondsp.overflow<saturate>, rounding = #ondsp.rounding<nearest_even>} : (!ondsp.acc<storage = i40, frac = 30, signed, update_overflow = wrap>) -> i16
      memref.store %2, %alloc[%i, %j] : memref<4x3xi16>
    }
  }
  memref.dealloc %packed : memref<3x16xi16>
  return %alloc : memref<4x3xi16>
}

memref.global "private" constant @row0 : memref<8xi16> = dense<[32767, 32767, 32767, 32767, 32767, 32767, 32767, 32767]>
memref.global "private" constant @row1 : memref<8xi16> = dense<[32138, 27245, 18204, 6392, -6392, -18204, -27245, -32138]>
memref.global "private" constant @row2 : memref<8xi16> = dense<[30273, 12539, -12539, -30273, -30273, -12539, 12539, 30273]>
memref.global "private" constant @row3 : memref<8xi16> = dense<[27245, -6392, -32138, -18204, 18204, 32138, 6392, -27245]>

// Four unrolled constant-row outputs at consecutive positions become one
// block: lane l carries row l, every row certifies, so the lanes wrap and pairs
// of terms accumulate in i32.
// CHECK-LABEL: func.func @constant_rows
// CHECK: ondsp.acc_zero : <storage = i40, frac = 30, signed, update_overflow = wrap, lanes = 4>
// CHECK: ondsp.acc_zero : <storage = i32, frac = 30, signed, update_overflow = wrap, lanes = 4>
// CHECK: arith.constant dense<[32767, 32138, 30273, 27245]> : vector<4xi16>
// CHECK: memref.load %{{.*}}[%{{.*}}] : memref<8xi16>
// CHECK-COUNT-2: ondsp.mac
// CHECK: ondsp.acc_export {{.*}} -> vector<4xi32>
// CHECK: ondsp.acc_add_term
// CHECK: ondsp.acc_export {{.*}} -> vector<4xi64>
// CHECK: ondsp.round_shift {{.*}} : (vector<4xi64>) -> vector<4xi16>
// CHECK: vector.store %{{.*}} : memref<4xi16>, vector<4xi16>
// CHECK-NOT: ondsp.reduce_mac
func.func @constant_rows(%input: memref<8xi16>) -> memref<4xi16> {
  %c3 = arith.constant 3 : index
  %c2 = arith.constant 2 : index
  %c1 = arith.constant 1 : index
  %c0 = arith.constant 0 : index
  %alloc = memref.alloc() {alignment = 64 : i64} : memref<4xi16>
  %r0 = memref.get_global @row0 : memref<8xi16>
  %r1 = memref.get_global @row1 : memref<8xi16>
  %r2 = memref.get_global @row2 : memref<8xi16>
  %r3 = memref.get_global @row3 : memref<8xi16>
  %zero = ondsp.acc_zero : <storage = i40, frac = 30, signed, update_overflow = saturate>
  %0 = ondsp.reduce_mac %zero, %input, %r0 {numeric = #ondsp.fixed<signed, storage = i16, frac = 15>, product = #ondsp.product<full>} : (!ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>, memref<8xi16>, memref<8xi16>) -> !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>
  %1 = ondsp.acc_export %0 {dst = #ondsp.fixed<signed, storage = i64, frac = 30>, overflow = #ondsp.overflow<saturate>, rounding = #ondsp.rounding<nearest_even>} : (!ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>) -> i64
  %2 = ondsp.round_shift %1 {scale = #ondsp.scale<pre_shift_left = 0, post_shift_right = 19, rounding = nearest_even, overflow = saturate, saturate_to = i16>} : (i64) -> i16
  memref.store %2, %alloc[%c0] : memref<4xi16>
  %3 = ondsp.reduce_mac %zero, %input, %r1 {numeric = #ondsp.fixed<signed, storage = i16, frac = 15>, product = #ondsp.product<full>} : (!ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>, memref<8xi16>, memref<8xi16>) -> !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>
  %4 = ondsp.acc_export %3 {dst = #ondsp.fixed<signed, storage = i64, frac = 30>, overflow = #ondsp.overflow<saturate>, rounding = #ondsp.rounding<nearest_even>} : (!ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>) -> i64
  %5 = ondsp.round_shift %4 {scale = #ondsp.scale<pre_shift_left = 0, post_shift_right = 19, rounding = nearest_even, overflow = saturate, saturate_to = i16>} : (i64) -> i16
  memref.store %5, %alloc[%c1] : memref<4xi16>
  %6 = ondsp.reduce_mac %zero, %input, %r2 {numeric = #ondsp.fixed<signed, storage = i16, frac = 15>, product = #ondsp.product<full>} : (!ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>, memref<8xi16>, memref<8xi16>) -> !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>
  %7 = ondsp.acc_export %6 {dst = #ondsp.fixed<signed, storage = i64, frac = 30>, overflow = #ondsp.overflow<saturate>, rounding = #ondsp.rounding<nearest_even>} : (!ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>) -> i64
  %8 = ondsp.round_shift %7 {scale = #ondsp.scale<pre_shift_left = 0, post_shift_right = 19, rounding = nearest_even, overflow = saturate, saturate_to = i16>} : (i64) -> i16
  memref.store %8, %alloc[%c2] : memref<4xi16>
  %9 = ondsp.reduce_mac %zero, %input, %r3 {numeric = #ondsp.fixed<signed, storage = i16, frac = 15>, product = #ondsp.product<full>} : (!ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>, memref<8xi16>, memref<8xi16>) -> !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>
  %10 = ondsp.acc_export %9 {dst = #ondsp.fixed<signed, storage = i64, frac = 30>, overflow = #ondsp.overflow<saturate>, rounding = #ondsp.rounding<nearest_even>} : (!ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>) -> i64
  %11 = ondsp.round_shift %10 {scale = #ondsp.scale<pre_shift_left = 0, post_shift_right = 19, rounding = nearest_even, overflow = saturate, saturate_to = i16>} : (i64) -> i16
  memref.store %11, %alloc[%c3] : memref<4xi16>
  return %alloc : memref<4xi16>
}

// Three outputs never fill a width-4 block; the ordered reductions stay.
// CHECK-LABEL: func.func @constant_rows_short
// CHECK-COUNT-3: ondsp.reduce_mac
// CHECK-NOT: ondsp.mac
func.func @constant_rows_short(%input: memref<8xi16>) -> memref<3xi16> {
  %c2 = arith.constant 2 : index
  %c1 = arith.constant 1 : index
  %c0 = arith.constant 0 : index
  %alloc = memref.alloc() {alignment = 64 : i64} : memref<3xi16>
  %r0 = memref.get_global @row0 : memref<8xi16>
  %r1 = memref.get_global @row1 : memref<8xi16>
  %r2 = memref.get_global @row2 : memref<8xi16>
  %zero = ondsp.acc_zero : <storage = i40, frac = 30, signed, update_overflow = saturate>
  %0 = ondsp.reduce_mac %zero, %input, %r0 {numeric = #ondsp.fixed<signed, storage = i16, frac = 15>, product = #ondsp.product<full>} : (!ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>, memref<8xi16>, memref<8xi16>) -> !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>
  %1 = ondsp.acc_export %0 {dst = #ondsp.fixed<signed, storage = i16, frac = 15>, overflow = #ondsp.overflow<saturate>, rounding = #ondsp.rounding<nearest_even>} : (!ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>) -> i16
  memref.store %1, %alloc[%c0] : memref<3xi16>
  %3 = ondsp.reduce_mac %zero, %input, %r1 {numeric = #ondsp.fixed<signed, storage = i16, frac = 15>, product = #ondsp.product<full>} : (!ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>, memref<8xi16>, memref<8xi16>) -> !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>
  %4 = ondsp.acc_export %3 {dst = #ondsp.fixed<signed, storage = i16, frac = 15>, overflow = #ondsp.overflow<saturate>, rounding = #ondsp.rounding<nearest_even>} : (!ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>) -> i16
  memref.store %4, %alloc[%c1] : memref<3xi16>
  %6 = ondsp.reduce_mac %zero, %input, %r2 {numeric = #ondsp.fixed<signed, storage = i16, frac = 15>, product = #ondsp.product<full>} : (!ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>, memref<8xi16>, memref<8xi16>) -> !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>
  %7 = ondsp.acc_export %6 {dst = #ondsp.fixed<signed, storage = i16, frac = 15>, overflow = #ondsp.overflow<saturate>, rounding = #ondsp.rounding<nearest_even>} : (!ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>) -> i16
  memref.store %7, %alloc[%c2] : memref<3xi16>
  return %alloc : memref<3xi16>
}
