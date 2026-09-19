// RUN: ondrix-opt %s --convert-ondsp-fixed-to-scalar | FileCheck %s

// Every product is exact and the two cross terms of a component combine before
// the accumulator sees them, so the i33 term type is what makes the difference
// of two i32 products representable without wrapping.
func.func @cx_dot_q15(
    %real: !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>,
    %imag: !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>,
    %lhs: memref<64xi32>, %rhs: memref<64xi32>)
    -> (!ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>,
        !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>) {
  %re, %im = ondsp.cx_reduce_mac %real, %imag, %lhs, %rhs {
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    layout = #ondsp.cx_layout<packed_i16_imag_hi_real_lo>
  } : (!ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>,
       !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>,
       memref<64xi32>, memref<64xi32>)
      -> (!ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>,
          !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>)
  return %re, %im : !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>,
                    !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>
}

// CHECK-LABEL: func.func @cx_dot_q15
// CHECK: scf.for {{.*}} iter_args(%[[RE:.*]] = %{{.*}}, %[[IM:.*]] = %{{.*}}) -> (i64, i64)
// CHECK: memref.load
// CHECK: memref.load
// CHECK: arith.trunci {{.*}} : i32 to i16
// CHECK: arith.extsi {{.*}} : i16 to i33
// CHECK: arith.muli {{.*}} : i33
// CHECK: arith.subi {{.*}} : i33
// CHECK: arith.addi {{.*}} : i33
// CHECK: scf.yield {{.*}} : i64, i64

// The conjugate flips exactly one sign in each component, which is the whole
// difference between a dot product and a correlation.
func.func @cx_correlate_q15(
    %real: !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>,
    %imag: !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>,
    %lhs: memref<64xi32>, %rhs: memref<64xi32>)
    -> (!ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>,
        !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>) {
  %re, %im = ondsp.cx_reduce_mac %real, %imag, %lhs, %rhs {
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    layout = #ondsp.cx_layout<packed_i16_imag_hi_real_lo>,
    conjugate
  } : (!ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>,
       !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>,
       memref<64xi32>, memref<64xi32>)
      -> (!ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>,
          !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>)
  return %re, %im : !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>,
                    !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>
}

// CHECK-LABEL: func.func @cx_correlate_q15
// CHECK: arith.addi {{.*}} : i33
// CHECK: arith.subi {{.*}} : i33
