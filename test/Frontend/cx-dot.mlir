// RUN: ondrix-compile %S/Inputs/q15_cx_dot.ox | FileCheck %s --check-prefix=DOT
// RUN: ondrix-compile %S/Inputs/q15_cx_dot.ox | ondrix-opt --convert-ondrix-to-ondsp --convert-ondsp-to-ortumcore | FileCheck %s --check-prefix=TARGET
// RUN: ondrix-compile %S/Inputs/q15_cx_correlate.ox | FileCheck %s --check-prefix=CORR
// RUN: ondrix-compile %S/Inputs/q15_cx_correlate.ox | ondrix-opt --convert-ondrix-to-ondsp --convert-ondsp-fixed-to-scalar | FileCheck %s --check-prefix=CORR-SCALAR
// RUN: ondrix-compile %S/Inputs/q31_cx_dot.ox | FileCheck %s --check-prefix=Q31
// RUN: not ondrix-compile %S/Inputs/invalid_cx_dot_accumulator.ox 2>&1 | FileCheck %s --check-prefix=WIDTH
// RUN: not ondrix-compile %S/Inputs/invalid_cx_dot_q31_width.ox 2>&1 | FileCheck %s --check-prefix=Q31WIDTH
// RUN: not ondrix-compile %S/Inputs/invalid_cx_dot_tensor.ox 2>&1 | FileCheck %s --check-prefix=CONTAINER
// RUN: not ondrix-compile %S/Inputs/invalid_cx_dot_conjugate.ox 2>&1 | FileCheck %s --check-prefix=CONJUGATE

// A complex_q15 buffer is the packed container, and the scalar result is one
// packed value the two exported components are assembled into.
// DOT-LABEL: func.func @q15_cx_dot
// DOT-SAME: memref<64xi32>
// DOT-SAME: -> i32
// DOT: ondrix.cx_dot
// DOT-SAME: layout = #ondsp.cx_layout<packed_i16_imag_hi_real_lo>
// DOT-SAME: numeric = #ondsp.fixed<signed, storage = i16, frac = 15>
// DOT-SAME: !ondsp.acc<storage = i32, frac = 30, signed, update_overflow = saturate>
// DOT-NOT: conjugate
// DOT: ondsp.acc_export
// DOT: ondsp.acc_export
// DOT: arith.extui
// DOT: arith.shli
// DOT: arith.ori

// TARGET-LABEL: func.func @q15_cx_dot
// TARGET: ortumcore.cx_reduce_mac
// TARGET-NOT: ondsp.

// CORR-LABEL: func.func @q15_cx_correlate
// CORR: ondrix.cx_dot
// CORR-SAME: conjugate
// CORR-SAME: !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>

// The conjugate form adds where the plain one subtracts, which is the whole
// difference between the two kernels.
// CORR-SCALAR-LABEL: func.func @q15_cx_correlate
// CORR-SCALAR: arith.addi {{.*}} : i33
// CORR-SCALAR: arith.subi {{.*}} : i33

// The Q31 profile is the same contract one component width up: the container
// doubles with it and the accumulator carries the exact product frac.
// Q31-LABEL: func.func @q31_cx_dot
// Q31-SAME: memref<64xi64>
// Q31-SAME: -> i64
// Q31: ondrix.cx_dot
// Q31-SAME: conjugate
// Q31-SAME: layout = #ondsp.cx_layout<packed_i32_imag_hi_real_lo>
// Q31-SAME: numeric = #ondsp.fixed<signed, storage = i32, frac = 31>
// Q31-SAME: !ondsp.acc<storage = i64, frac = 62, signed, update_overflow = saturate>

// WIDTH: error: cx_dot requires an explicit exact accumulator of width 32 or 40
// Q31WIDTH: error: cx_dot requires an explicit exact accumulator of width 64
// CONTAINER: error: cx_dot requires two rank-1 complex_q15 or complex_q31 buffer parameters
// CONJUGATE: error: a complex reduction accepts only conjugate=true or conjugate=false
