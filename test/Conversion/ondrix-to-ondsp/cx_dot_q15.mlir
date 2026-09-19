// RUN: ondrix-opt %s --convert-ondrix-to-ondsp | FileCheck %s

!acc = !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>

// The algorithm layer seeds both components; the conjugate flag is the only
// difference between the dot product and the correlation below.
func.func @cx_dot(%lhs: memref<64xi32>, %rhs: memref<64xi32>) -> (!acc, !acc) {
  %re, %im = ondrix.cx_dot %lhs, %rhs {
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    layout = #ondsp.cx_layout<packed_i16_imag_hi_real_lo>
  } : (memref<64xi32>, memref<64xi32>) -> (!acc, !acc)
  return %re, %im : !acc, !acc
}

// CHECK-LABEL: func.func @cx_dot
// CHECK: %[[RE:.*]] = ondsp.acc_zero
// CHECK: %[[IM:.*]] = ondsp.acc_zero
// CHECK: ondsp.cx_reduce_mac %[[RE]], %[[IM]]
// CHECK-NOT: conjugate

func.func @cx_correlate(%lhs: memref<64xi32>, %rhs: memref<64xi32>) -> (!acc, !acc) {
  %re, %im = ondrix.cx_dot %lhs, %rhs {
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    layout = #ondsp.cx_layout<packed_i16_imag_hi_real_lo>,
    conjugate
  } : (memref<64xi32>, memref<64xi32>) -> (!acc, !acc)
  return %re, %im : !acc, !acc
}

// CHECK-LABEL: func.func @cx_correlate
// CHECK: ondsp.cx_reduce_mac {{.*}}conjugate
