// RUN: ondrix-opt %s --convert-ondsp-fixed-to-scalar | FileCheck %s

!acc = !ondsp.acc<storage = i64, frac = 62, signed, update_overflow = saturate>

// Two Q31 components multiply to 62 bits and a component's cross terms are a
// DIFFERENCE of two of them, which reaches 2^63: the i65 term carrier is what
// makes that representable, and it is why no i64 spelling of this reduction
// exists. The i66 update carrier is the accumulator's own clamp.
func.func @cx_dot_q31(%real: !acc, %imag: !acc,
                      %lhs: memref<8xi64>, %rhs: memref<8xi64>) -> (!acc, !acc) {
  %re, %im = ondsp.cx_reduce_mac %real, %imag, %lhs, %rhs {
    numeric = #ondsp.fixed<signed, storage = i32, frac = 31>,
    layout = #ondsp.cx_layout<packed_i32_imag_hi_real_lo>
  } : (!acc, !acc, memref<8xi64>, memref<8xi64>) -> (!acc, !acc)
  return %re, %im : !acc, !acc
}

// CHECK-LABEL: func.func @cx_dot_q31
// CHECK: scf.for {{.*}} -> (i64, i64)
// CHECK: arith.trunci {{.*}} : i64 to i32
// CHECK: arith.extsi {{.*}} : i32 to i65
// CHECK: arith.muli {{.*}} : i65
// CHECK: arith.subi {{.*}} : i65
// CHECK: arith.addi {{.*}} : i65
// CHECK: arith.extsi {{.*}} : i64 to i66
