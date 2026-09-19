// RUN: ondrix-compile %S/Inputs/q15_cx_dot.ox | FileCheck %s --check-prefix=DOT
// RUN: ondrix-compile %S/Inputs/q15_cx_dot.ox | ondrix-opt --convert-ondrix-to-ondsp --convert-ondsp-to-ortumcore | FileCheck %s --check-prefix=TARGET
// RUN: ondrix-compile %S/Inputs/q15_cx_correlate.ox | FileCheck %s --check-prefix=CORR
// RUN: ondrix-compile %S/Inputs/q15_cx_correlate.ox | ondrix-opt --convert-ondrix-to-ondsp --convert-ondsp-fixed-to-scalar | FileCheck %s --check-prefix=CORR-SCALAR
// RUN: not ondrix-compile %S/Inputs/invalid_cx_dot_accumulator.ox 2>&1 | FileCheck %s --check-prefix=WIDTH
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

// WIDTH: error: cx_dot requires an explicit exact accumulator of width 32 or 40
// CONTAINER: error: cx_dot requires two rank-1 complex_q15 buffer parameters
// CONJUGATE: error: cx_dot accepts only conjugate=true or conjugate=false
